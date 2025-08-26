from typing import Union, Dict, List, Any, Tuple, Optional
import pandas as pd
import json
from difflib import SequenceMatcher
from functools import lru_cache
import collections
from collections import deque

from ..factory.entity_factory import EntityFactory
from ..models.providers import ModelProvider
from ..utils.config_loader import load_config
from .chunker import Chunker
from .evaluator import Evaluator


class TokenCache:
    """Simple token cache for frequently accessed chunks"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_order = deque(maxlen=max_size)
        self.max_size = max_size
    
    def get(self, text: str) -> Optional[List[str]]:
        if text in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(text)
            self.access_order.append(text)
            return self.cache[text]
        return None
    
    def put(self, text: str, tokens: List[str]):
        if len(self.cache) >= self.max_size and text not in self.cache:
            # Remove least recently used
            if self.access_order:
                oldest = self.access_order.popleft()
                del self.cache[oldest]
        
        self.cache[text] = tokens
        if text in self.access_order:
            self.access_order.remove(text)
        self.access_order.append(text)


class Extractor:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)
        self.mode = self.config["mode"]
        self.entity_factory = EntityFactory(
            extraction_dir=self.config.get("extraction_dir", "config/extraction"),
            evaluation_dir=self.config.get("evaluation_dir", "config/evaluation")
        )
        self.model = ModelProvider(self.config.get("model"))
        self.chunker = Chunker(config_path)
        self.evaluator = Evaluator(
            eval_dir=self.config.get("evaluation_dir", "config/evaluation"),
            settings_path=config_path
        )
        # Initialize token cache
        self.token_cache = TokenCache(max_size=100)
        
    def extract(self, df: Union[pd.DataFrame, 'SparkDF']) -> Tuple[Union[pd.DataFrame, 'SparkDF'], Dict]:
        """Extract entities from dataframe"""
        # Sample if configured
        sample_size = self.config.get("sample_size")
        if sample_size:
            df = self._sample(df, sample_size)
            
        entities = self.entity_factory.get_entities()
        
        if self.mode == "databricks":
            return self._extract_spark(df, entities)
        return self._extract_pandas(df, entities)
    
    def _extract_pandas(self, df: pd.DataFrame, entities: list) -> Tuple[pd.DataFrame, Dict]:
        """Extract using pandas"""
        extraction_results = {}
        
        for entity in entities:
            col_name = f"{entity.name}_extracted"
            df[col_name] = df.apply(
                lambda row: self._process_row(row, entity), axis=1
            )
            extraction_results[entity.name] = df[col_name].tolist()
            
            # Run evaluations for this entity if enabled
            if self.config.get("evaluation", {}).get("enabled", True):
                active_evals = entity.get_active_evaluations()
                for eval_name, eval_config in active_evals.items():
                    eval_col = f"{entity.name}_{eval_name}"
                    df[eval_col] = df.apply(
                        lambda row: self.evaluator._evaluate_extraction(
                            row, row[col_name], eval_config, entity.name
                        ), axis=1
                    )
        
        return df, extraction_results
    
    def _extract_spark(self, df: 'SparkDF', entities: list) -> Tuple['SparkDF', Dict]:
        """Extract using Spark DataFrame"""
        from pyspark.sql.functions import udf, col
        from pyspark.sql.types import StringType
        
        # Configure partitions
        spark_config = self.config.get("spark", {})
        if spark_config.get("use_partitions", True):
            num_partitions = spark_config.get("num_partitions", 200)
            df = df.repartition(num_partitions)
        
        extraction_results = {}
        
        for entity in entities:
            # Create extraction UDF
            def extract_entity(paragraphs, tables):
                row = {"paragraphs": paragraphs, "tables": tables}
                result = self._process_row(row, entity)
                return json.dumps(result)
            
            extract_udf = udf(extract_entity, StringType())
            col_name = f"{entity.name}_extracted"
            df = df.withColumn(col_name, extract_udf(col("paragraphs"), col("tables")))
            
            # Collect results for summary
            extraction_results[entity.name] = df.select(col_name).rdd.map(lambda x: x[0]).collect()
            
            # Run evaluations if enabled
            if self.config.get("evaluation", {}).get("enabled", True):
                active_evals = entity.get_active_evaluations()
                for eval_name, eval_config in active_evals.items():
                    eval_col = f"{entity.name}_{eval_name}"
                    
                    def evaluate(extraction_json, paragraphs):
                        extractions = json.loads(extraction_json) if extraction_json else []
                        row = {"paragraphs": paragraphs}
                        return self.evaluator._evaluate_extraction(
                            row, extractions, eval_config, entity.name
                        )
                    
                    eval_udf = udf(evaluate, StringType())
                    df = df.withColumn(eval_col, eval_udf(col(col_name), col("paragraphs")))
        
        return df, extraction_results
    
    def _process_row(self, row: Dict, entity: Any) -> List[Dict]:
        """Process a single row"""
        paragraphs = row.get("paragraphs", [])
        tables = row.get("tables")
        
        # Chunk content
        chunks = self.chunker.chunk(paragraphs, tables)
        if not chunks:
            return []
        
        # Check batching
        batch_config = self.config.get("model", {}).get("batching", {})
        threshold = batch_config.get("threshold", 10)
        use_batch = len(chunks) >= threshold and batch_config.get("enabled", False)
        
        results = []
        if use_batch:
            # Batch processing
            prompts = [entity.format_prompt(chunk) for chunk in chunks]
            responses = self.model.batch_infer(prompts, model_name=entity.model_override)
            
            for response, chunk in zip(responses, chunks):
                extraction = self._parse_response(response)
                matched = self._progressive_match(extraction, chunk)
                if matched:
                    results.append(matched)
        else:
            # Single processing
            for chunk in chunks:
                prompt = entity.format_prompt(chunk)
                response = self.model.infer(prompt, model_name=entity.model_override)
                extraction = self._parse_response(response)
                matched = self._progressive_match(extraction, chunk)
                if matched:
                    results.append(matched)
        
        return results
    
    def _parse_response(self, response: str) -> Dict:
        """Parse model response"""
        # Handle markdown code blocks
        if "```" in response:
            # Remove markdown formatting
            response = response.replace("```json", "").replace("```", "")
            # Extract JSON content
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                response = response[start:end]
        
        response = response.strip()
        
        try:
            parsed = json.loads(response)
            # For signing_party responses without text field, create one
            if "parties" in parsed and "text" not in parsed:
                party_names = [p.get("party_name", "") for p in parsed["parties"] if isinstance(p, dict)]
                if party_names:
                    parsed["text"] = ", ".join(party_names)
            return parsed
        except json.JSONDecodeError:
            return {"text": response}
    
    def _progressive_match(self, extraction: Dict, text: str) -> Optional[Dict]:
        """Progressive matching: exact -> fuzzy with increasing thresholds"""
        matching_config = self.config.get("extraction", {}).get("matching", {})
        
        # Handle responses with "extractions" array (like signing_party)
        if "extractions" in extraction and isinstance(extraction["extractions"], list):
            if not extraction["extractions"]:
                return None
            extraction["match_type"] = "extracted"
            return extraction
        
        # Stage 1: Cheap exact match
        extract_text = extraction.get("text", "")
        if extract_text and extract_text in text:
            extraction["match_type"] = "exact"
            return extraction
        
        # Stage 2: Return None if exact_only is configured
        if matching_config.get("exact_only", False):
            return None
        
        # Stage 3: Token-based fuzzy matching (only when needed)
        if not extract_text:
            return None
        
        # Get or create tokenized versions
        source_tokens = self._get_tokens(text)
        extract_tokens = self._get_tokens(extract_text)
        
        if not extract_tokens:
            return None
        
        # Try progressively relaxed thresholds
        fuzzy_passes = matching_config.get("fuzzy_passes", 1)
        for pass_num in range(1, fuzzy_passes + 1):
            threshold_key = f"fuzzy_threshold_{pass_num}"
            threshold = matching_config.get(threshold_key, matching_config.get("fuzzy_threshold", 0.75))
            
            # Use token-based matching for better performance
            ratio = self._token_based_match(extract_tokens, source_tokens, threshold)
            if ratio >= threshold:
                extraction["match_score"] = ratio
                extraction["match_type"] = "fuzzy"
                extraction["match_pass"] = f"fuzzy_{pass_num}"
                return extraction
        
        return None
    
    def _get_tokens(self, text: str) -> List[str]:
        """Get tokens with caching"""
        # Check cache first
        cached = self.token_cache.get(text)
        if cached is not None:
            return cached
        
        # Tokenize (simple word splitting for now)
        tokens = [self._normalize_token(t) for t in text.lower().split()]
        
        # Cache the result
        self.token_cache.put(text, tokens)
        return tokens
    
    @lru_cache(maxsize=10000)
    def _normalize_token(self, token: str) -> str:
        """Normalize token with caching (lowercasing, light stemming)"""
        token = token.lower().strip()
        # Light stemming: remove trailing 's' for plurals
        if len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
            token = token[:-1]
        return token
    
    def _token_based_match(self, extract_tokens: List[str], source_tokens: List[str], threshold: float) -> float:
        """Efficient token-based matching with optimization"""
        if not extract_tokens or not source_tokens:
            return 0.0
        
        len_e = len(extract_tokens)
        len_s = len(source_tokens)
        
        # Quick check: if source is too short, skip
        if len_s < len_e * threshold:
            return 0.0
        
        # Use sliding window for efficiency
        best_ratio = 0.0
        min_overlap = int(len_e * threshold)
        
        # Pre-compute extraction token counts for fast intersection
        extract_counts = collections.Counter(extract_tokens)
        
        # Try different window sizes
        for window_size in range(len_e, min(len_e * 2, len_s) + 1):
            if window_size > len_s:
                break
            
            # Initialize sliding window
            window_deque = deque(source_tokens[:window_size])
            window_counts = collections.Counter(window_deque)
            
            for start_idx in range(len_s - window_size + 1):
                # Fast pre-check using counter intersection
                overlap = (extract_counts & window_counts).total()
                if overlap >= min_overlap:
                    # Only compute expensive SequenceMatcher if pre-check passes
                    matcher = SequenceMatcher(None, extract_tokens, list(window_deque))
                    ratio = matcher.ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        # Early termination if we found a great match
                        if ratio >= 0.95:
                            return ratio
                
                # Slide window
                if start_idx + window_size < len_s:
                    # Remove leftmost token
                    old_token = window_deque.popleft()
                    window_counts[old_token] -= 1
                    if window_counts[old_token] == 0:
                        del window_counts[old_token]
                    
                    # Add new rightmost token
                    new_token = source_tokens[start_idx + window_size]
                    window_deque.append(new_token)
                    window_counts[new_token] += 1
        
        return best_ratio
    
    def _sample(self, df: Union[pd.DataFrame, 'SparkDF'], size: int) -> Union[pd.DataFrame, 'SparkDF']:
        """Sample dataframe"""
        if isinstance(df, pd.DataFrame):
            return df.head(size)
        return df.limit(size)