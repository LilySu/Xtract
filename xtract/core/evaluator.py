import json
from pathlib import Path
from typing import Dict, List, Any, Union
import pandas as pd
from difflib import SequenceMatcher

from ..utils.config_loader import load_yaml, load_config
from ..models.providers import ModelProvider


class Evaluator:
    def __init__(
        self,
        eval_dir: str = "config/evaluation",
        settings_path: str = "config/settings.yaml",
    ):
        self.settings = load_config(settings_path)
        self.eval_configs = self._load_eval_configs(eval_dir)
        self.model = None
        if self._needs_llm():
            self.model = ModelProvider(self.settings.get("model"))

    def _load_eval_configs(self, directory: str) -> Dict[str, Dict]:
        """Load evaluation configurations"""
        configs = {}
        eval_path = Path(directory)

        if eval_path.exists():
            for yaml_file in eval_path.glob("*.yaml"):
                config = load_yaml(yaml_file)
                if config.get("enabled", True):
                    configs[yaml_file.stem] = config

        return configs

    def _needs_llm(self) -> bool:
        """Check if any evaluation needs LLM"""
        llm_methods = [
            "context_complexity",
            "hallucination_detection",
            "completeness_check",
            "consistency_check",
            "relevance_score",
        ]
        return any(
            cfg.get("method") in llm_methods for cfg in self.eval_configs.values()
        )

    def evaluate(
        self, df: pd.DataFrame, extraction_results: Dict[str, List]
    ) -> pd.DataFrame:
        """Evaluate all extractions"""
        for eval_name, config in self.eval_configs.items():
            if not config.get("enabled", True):
                continue

            for entity_name, extractions in extraction_results.items():
                col_name = f"{entity_name}_{eval_name}"
                df[col_name] = df.apply(
                    lambda row: self._evaluate_extraction(
                        row, extractions, config, entity_name
                    ),
                    axis=1,
                )

        return df

    def _evaluate_extraction(
        self,
        row: Union[pd.Series, Dict],
        extractions: Any,
        config: Dict,
        entity_name: str,
    ) -> Any:
        """Apply single evaluation"""
        method = config.get("method")

        # Handle extraction format
        if isinstance(extractions, str):
            try:
                extractions = json.loads(extractions)
            except json.JSONDecodeError:
                extractions = []

        # Get extraction for current row if needed
        if isinstance(row, pd.Series) and f"{entity_name}_extracted" in row:
            extractions = row[f"{entity_name}_extracted"]

        # Rule-based evaluations
        if method == "exact_match":
            return self._exact_match(row, extractions, config)
        elif method == "fuzzy_match" or method == "semantic_similarity":
            return self._fuzzy_match(row, extractions, config)

        # LLM-based evaluations
        elif self.model and method in [
            "context_complexity",
            "hallucination_detection",
            "completeness_check",
            "consistency_check",
            "relevance_score",
        ]:
            return self._llm_evaluate(row, extractions, config)

        return None

    def _exact_match(
        self, row: Union[pd.Series, Dict], extractions: List, config: Dict
    ) -> str:
        """Check exact match"""
        source_text = self._get_source_text(row)

        if not isinstance(extractions, list):
            extractions = [extractions] if extractions else []

        for extraction in extractions:
            if isinstance(extraction, dict):
                text = extraction.get("text", "")
            else:
                text = str(extraction)

            if text and text in source_text:
                return "matched"

        return "no_match"

    def _fuzzy_match(
        self, row: Union[pd.Series, Dict], extractions: List, config: Dict
    ) -> str:
        """Fuzzy matching"""
        threshold = config.get("threshold", 0.85)
        source_text = self._get_source_text(row)

        if not isinstance(extractions, list):
            extractions = [extractions] if extractions else []

        for extraction in extractions:
            if isinstance(extraction, dict):
                text = extraction.get("text", "")
            else:
                text = str(extraction)

            if text:
                ratio = SequenceMatcher(None, text, source_text).ratio()
                if ratio >= threshold:
                    return "matched"

        return "no_match"

    def _llm_evaluate(
        self, row: Union[pd.Series, Dict], extractions: List, config: Dict
    ) -> str:
        """LLM-based evaluation"""
        prompt_template = config.get("prompt")
        if not prompt_template:
            return "no_prompt_configured"

        # Prepare context
        max_context = self.settings.get("evaluation", {}).get("max_eval_context", 2000)
        source_text = self._get_source_text(row)[:max_context]

        context_vars = {
            "source": source_text,
            "extraction": self._format_extractions(extractions),
            "extractions": self._format_extractions(extractions),
            "entity_type": config.get("entity_type", "entity"),
            "context": config.get("context", "document"),
        }

        # Format prompt
        formatted_prompt = prompt_template.format(**context_vars)

        # Get response
        response = self.model.infer(formatted_prompt)

        # Parse response
        expected_outputs = config.get("expected_outputs", [])
        response_lower = response.lower()

        for output in expected_outputs:
            if output.lower() in response_lower:
                return output

        return config.get("default_output", "unknown")

    def _get_source_text(self, row: Union[pd.Series, Dict]) -> str:
        """Extract source text"""
        if isinstance(row, dict):
            paragraphs = row.get("paragraphs", [])
        else:
            paragraphs = row.get("paragraphs", [])

        if isinstance(paragraphs, list):
            texts = []
            for p in paragraphs:
                if isinstance(p, dict):
                    texts.append(p.get("text", ""))
                else:
                    texts.append(str(p))
            return " ".join(texts)

        return str(paragraphs)

    def _format_extractions(self, extractions: List) -> str:
        """Format extractions for prompt"""
        if not extractions:
            return "No extractions"

        if not isinstance(extractions, list):
            extractions = [extractions]

        formatted = []
        for ext in extractions:
            if isinstance(ext, dict):
                text = ext.get("text", "")
                value = ext.get("value", "")
                if value:
                    formatted.append(f"- {text}: {value}")
                else:
                    formatted.append(f"- {text}")
            else:
                formatted.append(f"- {ext}")

        return "\n".join(formatted) if formatted else "No extractions"
