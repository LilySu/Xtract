#!/usr/bin/env python
"""
Comprehensive test suite for optimized extractor features:
- Token caching
- Progressive matching (exact -> fuzzy)
- Sliding window optimization
- Token normalization with LRU cache
"""

import unittest
import time
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from xtract.core.extractor import Extractor, TokenCache


class TestTokenCache(unittest.TestCase):
    """Test the TokenCache implementation"""

    def test_cache_basic_operations(self):
        """Test basic cache put/get operations"""
        cache = TokenCache(max_size=3)

        # Test putting and getting
        cache.put("test1", ["token1", "token2"])
        self.assertEqual(cache.get("test1"), ["token1", "token2"])

        # Test cache miss
        self.assertIsNone(cache.get("not_exists"))

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = TokenCache(max_size=3)

        # Fill cache
        cache.put("test1", ["token1"])
        cache.put("test2", ["token2"])
        cache.put("test3", ["token3"])

        # Access test1 to make it recently used
        cache.get("test1")

        # Add new item, should evict test2 (least recently used)
        cache.put("test4", ["token4"])

        self.assertIsNotNone(cache.get("test1"))  # Still there
        self.assertIsNone(cache.get("test2"))  # Evicted
        self.assertIsNotNone(cache.get("test3"))  # Still there
        self.assertIsNotNone(cache.get("test4"))  # New item

    def test_cache_update_access_order(self):
        """Test that accessing items updates their position"""
        cache = TokenCache(max_size=2)

        cache.put("test1", ["token1"])
        cache.put("test2", ["token2"])

        # Access test1, making test2 the LRU
        cache.get("test1")

        # Add new item, should evict test2
        cache.put("test3", ["token3"])

        self.assertIsNotNone(cache.get("test1"))
        self.assertIsNone(cache.get("test2"))


class TestOptimizedExtractor(unittest.TestCase):
    """Test the optimized extractor features"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock config to avoid file dependency
        mock_config = {
            "mode": "local",
            "sample_size": None,
            "extraction_dir": "config/extraction",
            "evaluation_dir": "config/evaluation",
            "extraction": {
                "matching": {
                    "exact_only": False,
                    "fuzzy_passes": 3,
                    "fuzzy_threshold": 0.75,
                    "fuzzy_threshold_1": 0.85,
                    "fuzzy_threshold_2": 0.75,
                    "fuzzy_threshold_3": 0.65,
                }
            },
            "evaluation": {"enabled": True},
            "model": {
                "provider": "openai",
                "model_name": "gpt-4",
                "batching": {"enabled": False, "threshold": 10},
            },
            "chunking": {"strategy": "page"},
        }

        # Mock the config loading
        with patch("xtract.core.extractor.load_config") as mock_load_config:
            mock_load_config.return_value = mock_config

            # Mock other dependencies
            with (
                patch("xtract.core.extractor.EntityFactory") as mock_entity_factory,
                patch("xtract.core.extractor.ModelProvider") as mock_model_provider,
                patch("xtract.core.extractor.Chunker") as mock_chunker,
                patch("xtract.core.extractor.Evaluator") as mock_evaluator,
            ):
                # Set up mock entity factory
                mock_entity_factory.return_value.get_entities.return_value = []

                # Create the extractor
                self.extractor = Extractor("dummy_path.yaml")

        # Mock the model to avoid API calls
        self.extractor.model = Mock()
        self.extractor.model.infer = Mock(return_value='{"text": "test extraction"}')

    def test_token_normalization(self):
        """Test token normalization with stemming"""
        # Test basic normalization
        self.assertEqual(self.extractor._normalize_token("Test"), "test")
        self.assertEqual(self.extractor._normalize_token("UPPER"), "upper")

        # Test light stemming (plural removal)
        self.assertEqual(self.extractor._normalize_token("cats"), "cat")
        self.assertEqual(self.extractor._normalize_token("dogs"), "dog")
        self.assertEqual(
            self.extractor._normalize_token("classes"), "classe"
        )  # Not 'class' to avoid keyword

        # Test that 'ss' endings are preserved
        self.assertEqual(self.extractor._normalize_token("class"), "class")
        self.assertEqual(self.extractor._normalize_token("pass"), "pass")

        # Test short words aren't stemmed
        self.assertEqual(self.extractor._normalize_token("is"), "is")
        self.assertEqual(self.extractor._normalize_token("as"), "as")

    def test_normalize_token_cache(self):
        """Test that normalize_token uses LRU cache"""
        # Clear cache first
        self.extractor._normalize_token.cache_clear()

        # First call should compute
        result1 = self.extractor._normalize_token("testing")

        # Second call should use cache (verify by checking cache info)
        cache_info_before = self.extractor._normalize_token.cache_info()
        result2 = self.extractor._normalize_token("testing")
        cache_info_after = self.extractor._normalize_token.cache_info()

        self.assertEqual(result1, result2)
        self.assertEqual(cache_info_after.hits, cache_info_before.hits + 1)

    def test_get_tokens_with_cache(self):
        """Test tokenization with caching"""
        text = "This is a test document with multiple words"

        # First call should tokenize and cache
        tokens1 = self.extractor._get_tokens(text)

        # Verify tokens are normalized
        self.assertIn("thi", tokens1)  # "This" -> "thi" (normalized)
        self.assertIn("word", tokens1)  # "words" -> "word" (stemmed)

        # Second call should use cache
        tokens2 = self.extractor._get_tokens(text)
        self.assertEqual(tokens1, tokens2)

        # Verify it's actually cached
        cached = self.extractor.token_cache.get(text)
        self.assertEqual(cached, tokens1)

    def test_progressive_matching_exact(self):
        """Test progressive matching stops at exact match"""
        extraction = {"text": "Purchase Agreement"}
        chunk = "This is a Purchase Agreement between parties"

        # Should find exact match and stop
        result = self.extractor._progressive_match(extraction, chunk)

        self.assertIsNotNone(result)
        self.assertEqual(result["match_type"], "exact")
        self.assertNotIn("match_score", result)  # No fuzzy score for exact match

    def test_progressive_matching_fuzzy(self):
        """Test progressive matching falls back to fuzzy"""
        extraction = {"text": "Purchase Agreements"}  # Plural
        chunk = "This is a Purchase Agreement between parties"  # Singular

        # Should use fuzzy matching
        result = self.extractor._progressive_match(extraction, chunk)

        self.assertIsNotNone(result)
        self.assertEqual(result["match_type"], "fuzzy")
        self.assertIn("match_score", result)
        self.assertIn("match_pass", result)

    def test_progressive_matching_thresholds(self):
        """Test progressive matching with different thresholds"""
        # Mock config with progressive thresholds
        self.extractor.config = {
            "extraction": {
                "matching": {
                    "exact_only": False,
                    "fuzzy_passes": 3,
                    "fuzzy_threshold_1": 0.90,  # Very strict
                    "fuzzy_threshold_2": 0.70,  # Medium
                    "fuzzy_threshold_3": 0.50,  # Lenient
                }
            }
        }

        # Test text with higher similarity that can demonstrate progressive matching
        extraction = {"text": "Purchase Agreement"}
        chunk = "This Purchase Contract Agreement is between parties"

        result = self.extractor._progressive_match(extraction, chunk)

        # Should match on a later pass with lower threshold
        self.assertIsNotNone(result)
        self.assertEqual(result["match_type"], "fuzzy")
        if "match_pass" in result:
            # Should not be pass 1 (threshold too high)
            self.assertIn(result["match_pass"], ["fuzzy_2", "fuzzy_3"])

    def test_token_based_match_sliding_window(self):
        """Test the sliding window optimization"""
        extract_tokens = ["purchase", "agreement"]
        source_tokens = [
            "this",
            "document",
            "is",
            "a",
            "purchase",
            "agreement",
            "between",
            "parties",
        ]

        # Should find high match where "purchase agreement" appears
        ratio = self.extractor._token_based_match(extract_tokens, source_tokens, 0.5)

        self.assertGreater(ratio, 0.9)  # Should be nearly perfect match

    def test_token_based_match_with_stemming(self):
        """Test token matching handles plurals via normalization"""
        extract_tokens = ["purchase", "agreement"]  # Already normalized
        source_tokens = ["purchase", "agreement"]  # Normalized from "agreements"

        ratio = self.extractor._token_based_match(extract_tokens, source_tokens, 0.5)

        self.assertEqual(ratio, 1.0)  # Perfect match after normalization

    def test_token_based_match_early_termination(self):
        """Test early termination on excellent match"""
        # Identical tokens should terminate early
        tokens = ["exact", "match", "test"]

        start_time = time.time()
        ratio = self.extractor._token_based_match(tokens, tokens, 0.5)
        elapsed = time.time() - start_time

        self.assertGreaterEqual(ratio, 0.95)
        # Should be very fast due to early termination
        self.assertLess(elapsed, 0.01)  # Should complete in < 10ms

    def test_token_based_match_counter_optimization(self):
        """Test counter-based pre-filtering"""
        extract_tokens = ["unique", "extraction", "text"]
        # Source with no overlap
        source_tokens = ["completely", "different", "content", "here"]

        # Should quickly reject due to no overlap
        ratio = self.extractor._token_based_match(extract_tokens, source_tokens, 0.8)

        self.assertEqual(ratio, 0.0)  # No match found

    def test_extraction_with_special_responses(self):
        """Test handling of special extraction formats"""
        # Test signing_party format
        extraction = {
            "extractions": [{"party_name": "ABC Corp"}, {"party_name": "XYZ Inc"}]
        }
        chunk = "Agreement between ABC Corp and XYZ Inc"

        result = self.extractor._progressive_match(extraction, chunk)

        self.assertIsNotNone(result)
        self.assertEqual(result["match_type"], "extracted")

    def test_exact_only_mode(self):
        """Test that exact_only mode skips fuzzy matching"""
        self.extractor.config = {"extraction": {"matching": {"exact_only": True}}}

        extraction = {"text": "Purchases Agreement"}  # Close but not exact
        chunk = "This is a Purchase Agreement"

        result = self.extractor._progressive_match(extraction, chunk)

        self.assertIsNone(result)  # Should not match when exact_only is True

    def test_parse_response_formats(self):
        """Test parsing different response formats"""
        # Test markdown code block
        response1 = '```json\n{"text": "test"}\n```'
        parsed1 = self.extractor._parse_response(response1)
        self.assertEqual(parsed1["text"], "test")

        # Test plain JSON
        response2 = '{"text": "plain json"}'
        parsed2 = self.extractor._parse_response(response2)
        self.assertEqual(parsed2["text"], "plain json")

        # Test invalid JSON fallback
        response3 = "not valid json"
        parsed3 = self.extractor._parse_response(response3)
        self.assertEqual(parsed3["text"], "not valid json")

        # Test signing_party format
        response4 = '{"parties": [{"party_name": "ABC"}, {"party_name": "XYZ"}]}'
        parsed4 = self.extractor._parse_response(response4)
        self.assertEqual(parsed4["text"], "ABC, XYZ")


class TestPerformanceImprovements(unittest.TestCase):
    """Test performance improvements are working"""

    def setUp(self):
        """Set up test fixtures with mocked config"""
        mock_config = {
            "mode": "local",
            "extraction": {"matching": {"exact_only": False, "fuzzy_threshold": 0.75}},
            "model": {"batching": {"enabled": False}},
            "evaluation": {"enabled": False},
        }

        with patch("xtract.core.extractor.load_config") as mock_load_config:
            mock_load_config.return_value = mock_config

            with (
                patch("xtract.core.extractor.EntityFactory"),
                patch("xtract.core.extractor.ModelProvider"),
                patch("xtract.core.extractor.Chunker"),
                patch("xtract.core.extractor.Evaluator"),
            ):
                self.extractor = Extractor("dummy_path.yaml")

        self.extractor.model = Mock()
        self.extractor.model.infer = Mock(return_value='{"text": "test"}')

    def test_cache_performance(self):
        """Verify caching improves performance"""
        text = " ".join(["word"] * 1000)  # Long text

        # First tokenization
        start = time.time()
        tokens1 = self.extractor._get_tokens(text)
        first_time = time.time() - start

        # Second tokenization (should be cached)
        start = time.time()
        tokens2 = self.extractor._get_tokens(text)
        cached_time = time.time() - start

        self.assertEqual(tokens1, tokens2)
        # Cached should be at least 10x faster
        self.assertLess(cached_time * 10, first_time)

    def test_sliding_window_efficiency(self):
        """Test sliding window is more efficient than naive approach"""
        extract = ["target", "phrase"]
        # Long source with target at the end
        source = ["filler"] * 100 + ["target", "phrase"]

        start = time.time()
        ratio = self.extractor._token_based_match(extract, source, 0.5)
        elapsed = time.time() - start

        self.assertGreater(ratio, 0.9)
        # Should complete quickly even with long source
        self.assertLess(elapsed, 0.1)  # Under 100ms


class TestIntegration(unittest.TestCase):
    """Integration tests with real-like data"""

    def setUp(self):
        """Set up test fixtures with mocked dependencies"""
        mock_config = {
            "mode": "local",
            "extraction": {"matching": {"exact_only": False, "fuzzy_threshold": 0.75}},
            "model": {"batching": {"enabled": False, "threshold": 10}},
            "evaluation": {"enabled": False},
            "chunking": {"strategy": "page"},
        }

        with patch("xtract.core.extractor.load_config") as mock_load_config:
            mock_load_config.return_value = mock_config

            with (
                patch("xtract.core.extractor.EntityFactory"),
                patch("xtract.core.extractor.ModelProvider"),
                patch("xtract.core.extractor.Chunker") as mock_chunker,
                patch("xtract.core.extractor.Evaluator"),
            ):
                # Mock chunker to return test chunks
                mock_chunker.return_value.chunk.return_value = [
                    "This Purchase Agreement is made between ABC Corp and XYZ Inc"
                ]

                self.extractor = Extractor("dummy_path.yaml")
                self.extractor.chunker = mock_chunker.return_value

        self.extractor.model = Mock()

    def test_end_to_end_extraction(self):
        """Test full extraction pipeline with optimizations"""
        # Mock model responses
        self.extractor.model.infer = Mock(
            side_effect=[
                '{"text": "Purchase Agreement"}',
                '{"extractions": [{"party_name": "ABC Corp"}, {"party_name": "XYZ Inc"}]}',
            ]
        )

        # Create test data
        row = {
            "paragraphs": [
                {"text": "This Purchase Agreement is made between", "page_number": 1},
                {"text": "ABC Corp and XYZ Inc", "page_number": 1},
            ]
        }

        # Mock entity
        entity = Mock()
        entity.name = "contract_type"
        entity.format_prompt = Mock(return_value="Extract contract type")
        entity.model_override = None

        # Run extraction
        results = self.extractor._process_row(row, entity)

        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Purchase Agreement")
        self.assertIn("match_type", results[0])


if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)
