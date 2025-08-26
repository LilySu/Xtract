# tests/test_databricks_mode.py
import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test imports
import sys
sys.path.append('..')

from src.extraction.extractor import Extractor, SPARK_AVAILABLE
from src.utils.io import read_csv, save_results, convert_to_spark, convert_to_pandas


class TestDatabricksMode:
    """Test suite for Databricks/Spark functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'path': ['doc1.pdf', 'doc2.pdf'],
            'paragraphs': [
                json.dumps([
                    {'text': 'This is a test agreement between Party A and Party B.', 'page_number': 1},
                    {'text': 'The effective date is January 1, 2024.', 'page_number': 1}
                ]),
                json.dumps([
                    {'text': 'This contract is between Company X and Company Y.', 'page_number': 1},
                    {'text': 'The term shall be 5 years.', 'page_number': 2}
                ])
            ],
            'tables': [
                json.dumps([]),
                json.dumps([])
            ]
        })
    
    @pytest.fixture
    def databricks_config(self, tmp_path):
        """Create temporary Databricks configuration"""
        config = {
            'mode': 'databricks',
            'sample_size': None,
            'extraction_dir': 'config/extraction',
            'evaluation_dir': 'config/evaluation',
            'spark': {
                'use_partitions': True,
                'num_partitions': 4
            },
            'chunking': {
                'strategy': 'page',
                'max_chars': 1000,
                'overlap': 100
            },
            'extraction': {
                'matching': {
                    'exact_only': False,
                    'fuzzy_passes': 1,
                    'fuzzy_threshold': 0.75
                }
            },
            'evaluation': {
                'enabled': False  # Disable for testing
            },
            'model': {
                'provider': 'openai',
                'model_name': 'gpt-4o-mini',
                'temperature': 0.0,
                'max_tokens': 100,
                'batching': {
                    'enabled': False
                }
            }
        }
        
        # Write config to temp file
        config_file = tmp_path / "settings.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_file)
    
    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_spark_dataframe_creation(self, sample_data):
        """Test conversion from pandas to Spark DataFrame"""
        spark_df = convert_to_spark(sample_data)
        
        # Check that it's a Spark DataFrame
        assert hasattr(spark_df, 'toPandas')
        assert spark_df.count() == 2
        assert 'path' in spark_df.columns
        assert 'paragraphs' in spark_df.columns
    
    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_spark_to_pandas_conversion(self, sample_data):
        """Test conversion from Spark back to pandas"""
        spark_df = convert_to_spark(sample_data)
        pandas_df = convert_to_pandas(spark_df)
        
        assert isinstance(pandas_df, pd.DataFrame)
        assert len(pandas_df) == 2
        assert list(pandas_df.columns) == list(sample_data.columns)
    
    def test_databricks_mode_fallback(self, databricks_config, sample_data, tmp_path):
        """Test that Databricks mode falls back to pandas when Spark not available"""
        # Save sample data
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Mock model provider to avoid API calls
        with patch('src.extraction.extractor.ModelProvider') as mock_model:
            mock_model.return_value.infer.return_value = '{"text": "test extraction"}'
            
            # Create extractor with databricks config
            extractor = Extractor(databricks_config)
            
            # If Spark not available, should fall back to local mode
            if not SPARK_AVAILABLE:
                assert extractor.mode == "local"
            else:
                assert extractor.mode == "databricks"
    
    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_spark_extraction_udf(self, databricks_config, sample_data):
        """Test Spark UDF extraction functionality"""
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, udf
        from pyspark.sql.types import StringType
        
        spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
        
        # Convert to Spark DataFrame
        spark_df = spark.createDataFrame(sample_data)
        
        # Test extraction UDF
        def extract_test(paragraphs, tables):
            # Simulate extraction
            return json.dumps([{"text": "extracted", "value": "test"}])
        
        extract_udf = udf(extract_test, StringType())
        result_df = spark_df.withColumn("test_extracted", extract_udf(col("paragraphs"), col("tables")))
        
        assert "test_extracted" in result_df.columns
        results = result_df.collect()
        assert len(results) == 2
        
        spark.stop()
    
    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_spark_save_results(self, sample_data, tmp_path):
        """Test saving Spark DataFrame results"""
        spark_df = convert_to_spark(sample_data)
        
        # Test CSV save
        csv_output = tmp_path / "output.csv"
        save_results(spark_df, str(csv_output), mode="databricks")
        
        # Check file was created
        assert csv_output.exists() or (tmp_path / "output").exists()
    
    def test_databricks_environment_detection(self):
        """Test Databricks environment detection"""
        from src.utils.io import is_databricks_environment, get_dbfs_path
        
        # Should return False in normal environment
        assert not is_databricks_environment()
        
        # Test DBFS path conversion
        path = get_dbfs_path("/mnt/data/file.csv")
        assert path == "/mnt/data/file.csv"  # No change when not in Databricks
        
        # Mock Databricks environment
        with patch.dict('os.environ', {'DATABRICKS_RUNTIME_VERSION': '11.3'}):
            assert is_databricks_environment()
            path = get_dbfs_path("mnt/data/file.csv")
            assert path == "/dbfs/mnt/data/file.csv"
    
    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_full_spark_extraction_pipeline(self, databricks_config, sample_data, tmp_path):
        """Test complete extraction pipeline in Spark mode"""
        # Save sample data
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Create mock entity config
        entity_config = {
            'enabled': True,
            'prompt': 'Extract parties from: {content}',
            'context': {},
            'format': {},
            'evaluations': {}
        }
        
        entity_path = tmp_path / "config" / "extraction"
        entity_path.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(entity_path / "party.yaml", 'w') as f:
            yaml.dump(entity_config, f)
        
        # Update config with temp extraction dir
        with open(databricks_config, 'r') as f:
            config = yaml.safe_load(f)
        config['extraction_dir'] = str(entity_path)
        with open(databricks_config, 'w') as f:
            yaml.dump(config, f)
        
        # Mock the model provider
        with patch('src.extraction.extractor.ModelProvider') as mock_model:
            mock_model.return_value.infer.return_value = '{"text": "Party A and Party B"}'
            
            # Create extractor and run
            extractor = Extractor(databricks_config)
            
            # Load data in appropriate format
            if extractor.mode == "databricks":
                df = convert_to_spark(sample_data)
            else:
                df = sample_data
            
            # Run extraction
            result_df, extraction_results = extractor.extract(df)
            
            # Verify results
            assert result_df is not None
            assert 'party_extracted' in extraction_results or len(extraction_results) > 0


class TestSparkUDFIntegration:
    """Test Spark UDF integration with extraction logic"""
    
    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_udf_json_parsing(self):
        """Test JSON parsing within Spark UDFs"""
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType
        import json
        
        spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
        
        # Create test data
        data = [(json.dumps([{"text": "test1"}]),), (json.dumps([{"text": "test2"}]),)]
        df = spark.createDataFrame(data, ["json_col"])
        
        # Define UDF that parses JSON
        def parse_json_udf(json_str):
            try:
                data = json.loads(json_str) if json_str else []
                return str(len(data))
            except:
                return "0"
        
        parse_udf = udf(parse_json_udf, StringType())
        result_df = df.withColumn("parsed_count", parse_udf("json_col"))
        
        results = result_df.collect()
        assert results[0]["parsed_count"] == "1"
        assert results[1]["parsed_count"] == "1"
        
        spark.stop()
    
    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_partitioning_configuration(self, databricks_config):
        """Test that partitioning configuration is applied correctly"""
        from pyspark.sql import SparkSession
        import yaml
        
        spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
        
        # Load config
        with open(databricks_config, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config['spark']['use_partitions'] == True
        assert config['spark']['num_partitions'] == 4
        
        # Create test DataFrame
        data = [(i,) for i in range(100)]
        df = spark.createDataFrame(data, ["id"])
        
        # Apply partitioning as per config
        if config['spark']['use_partitions']:
            df = df.repartition(config['spark']['num_partitions'])
        
        assert df.rdd.getNumPartitions() == 4
        
        spark.stop()