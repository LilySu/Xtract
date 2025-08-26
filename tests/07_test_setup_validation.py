#!/usr/bin/env python
"""
Validation script to test different modes and configurations.
Run this to verify your setup is working correctly.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_environment():
    """Check environment variables and dependencies"""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✓ OPENAI_API_KEY is set ({len(openai_key)} chars)")
    else:
        print("✗ OPENAI_API_KEY is not set")
    
    # Check Azure credentials
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_endpoint:
        print(f"✓ Azure OpenAI credentials are set")
        print(f"  - Endpoint: {azure_endpoint}")
    else:
        print("✗ Azure OpenAI credentials not set (optional)")
    
    # Check PySpark
    try:
        import pyspark
        print(f"✓ PySpark is installed (version {pyspark.__version__})")
        spark_available = True
    except ImportError:
        print("✗ PySpark is not installed (optional, needed for Databricks mode)")
        spark_available = False
    
    # Check OpenAI package
    try:
        import openai
        print(f"✓ OpenAI package is installed (version {openai.__version__})")
        openai_available = True
    except ImportError:
        print("✗ OpenAI package is not installed - run: pip install openai")
        openai_available = False
    
    print("")
    return {
        'openai_key': bool(openai_key),
        'azure_configured': bool(azure_key and azure_endpoint),
        'spark_available': spark_available,
        'openai_available': openai_available
    }


def test_openai_provider(skip_if_no_key=True):
    """Test OpenAI provider"""
    print("=" * 60)
    print("TESTING OPENAI PROVIDER")
    print("=" * 60)
    
    if skip_if_no_key and not os.getenv("OPENAI_API_KEY"):
        print("⚠ Skipping - no OPENAI_API_KEY set")
        return False
    
    try:
        from src.models.providers import ModelProvider
        
        config = {
            'provider': 'openai',
            'model_name': 'gpt-4o-mini',
            'temperature': 0.0,
            'max_tokens': 50
        }
        
        provider = ModelProvider(config)
        print(f"Provider initialized in {'simple' if provider.simple_mode else 'router'} mode")
        
        # Test inference
        response = provider.infer("Say 'Hello, extraction framework is working!' and nothing else.")
        print(f"✓ OpenAI response: {response[:100]}...")
        
        # Test batch inference
        prompts = ["Count to 1", "Count to 2", "Count to 3"]
        responses = provider.batch_infer(prompts)
        print(f"✓ Batch inference completed: {len(responses)} responses")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenAI test failed: {str(e)}")
        return False


def test_azure_provider(skip_if_not_configured=True):
    """Test Azure OpenAI provider"""
    print("=" * 60)
    print("TESTING AZURE OPENAI PROVIDER")
    print("=" * 60)
    
    if skip_if_not_configured and not (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")):
        print("⚠ Skipping - Azure OpenAI not configured")
        return False
    
    try:
        from src.models.providers import AzureOpenAIProvider
        
        config = {
            'endpoint': os.getenv("AZURE_OPENAI_ENDPOINT"),
            'api_key': os.getenv("AZURE_OPENAI_API_KEY"),
            'api_version': '2024-10-21',
            'models': {
                'gpt-4o-mini': {
                    'deployment': 'gpt-4o-mini-deployment',  # Update with your deployment name
                    'temperature': 0.0,
                    'max_tokens': 50
                }
            }
        }
        
        provider = AzureOpenAIProvider('gpt-4o-mini', config)
        
        if provider.client:
            response = provider.infer("Say 'Hello from Azure!' and nothing else.")
            print(f"✓ Azure response: {response[:100]}...")
            return True
        else:
            print("✗ Azure client not initialized")
            return False
            
    except Exception as e:
        print(f"✗ Azure test failed: {str(e)}")
        return False


def test_databricks_mode():
    """Test Databricks/Spark mode"""
    print("=" * 60)
    print("TESTING DATABRICKS MODE")
    print("=" * 60)
    
    try:
        from src.extraction.extractor import Extractor, SPARK_AVAILABLE
        
        if not SPARK_AVAILABLE:
            print("✗ PySpark not available")
            print("  To enable Databricks mode, run: pip install pyspark")
            return False
        
        from src.utils.io import convert_to_spark, convert_to_pandas
        
        # Create test data
        test_df = pd.DataFrame({
            'path': ['test1.pdf', 'test2.pdf'],
            'paragraphs': [
                json.dumps([{'text': 'Test document 1', 'page_number': 1}]),
                json.dumps([{'text': 'Test document 2', 'page_number': 1}])
            ],
            'tables': [json.dumps([]), json.dumps([])]
        })
        
        # Convert to Spark
        spark_df = convert_to_spark(test_df)
        print(f"✓ Created Spark DataFrame with {spark_df.count()} rows")
        
        # Convert back to pandas
        pandas_df = convert_to_pandas(spark_df)
        print(f"✓ Converted back to pandas with {len(pandas_df)} rows")
        
        # Test with extractor (without actual extraction to avoid API calls)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'mode': 'databricks',
                'extraction_dir': 'config/extraction',
                'evaluation_dir': 'config/evaluation',
                'spark': {'use_partitions': True, 'num_partitions': 2},
                'chunking': {'strategy': 'page'},
                'model': {'provider': 'openai', 'model_name': 'gpt-4o-mini'},
                'evaluation': {'enabled': False}
            }
            import yaml
            yaml.dump(config, f)
            config_path = f.name
        
        extractor = Extractor(config_path)
        print(f"✓ Extractor initialized in {extractor.mode} mode")
        
        # Clean up
        os.unlink(config_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Databricks test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_extraction_pipeline():
    """Test a minimal extraction pipeline"""
    print("=" * 60)
    print("TESTING EXTRACTION PIPELINE")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ Skipping - no OPENAI_API_KEY set")
        return False
    
    try:
        from src.extraction.extractor import Extractor
        from src.factory.entity_factory import EntityFactory
        
        # Create temporary config
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings
            settings_path = Path(tmpdir) / "settings.yaml"
            settings = {
                'mode': 'local',
                'extraction_dir': str(Path(tmpdir) / 'extraction'),
                'evaluation_dir': str(Path(tmpdir) / 'evaluation'),
                'chunking': {'strategy': 'page'},
                'extraction': {'matching': {'exact_only': False}},
                'evaluation': {'enabled': False},
                'model': {
                    'provider': 'openai',
                    'model_name': 'gpt-4o-mini',
                    'temperature': 0.0,
                    'max_tokens': 100
                }
            }
            
            import yaml
            with open(settings_path, 'w') as f:
                yaml.dump(settings, f)
            
            # Create a test entity config
            extraction_dir = Path(tmpdir) / 'extraction'
            extraction_dir.mkdir()
            
            entity_config = {
                'enabled': True,
                'prompt': 'Extract any company names from: {content}\nReturn as JSON: {{"text": "company name"}}',
                'context': {},
                'evaluations': {}
            }
            
            with open(extraction_dir / 'company.yaml', 'w') as f:
                yaml.dump(entity_config, f)
            
            # Create test data
            test_df = pd.DataFrame({
                'path': ['test.pdf'],
                'paragraphs': [
                    [{'text': 'This agreement is between Acme Corp and Beta Industries.', 'page_number': 1}]
                ],
                'tables': [[]]
            })
            
            # Run extraction
            extractor = Extractor(str(settings_path))
            result_df, extraction_results = extractor.extract(test_df)
            
            print(f"✓ Extraction completed")
            print(f"  - Extracted columns: {[c for c in result_df.columns if 'extracted' in c]}")
            
            if 'company_extracted' in result_df.columns:
                extracted = result_df['company_extracted'].iloc[0]
                print(f"  - Sample extraction: {extracted}")
            
            return True
            
    except Exception as e:
        print(f"✗ Extraction pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n" + "=" * 60)
    print("DOCUMENT EXTRACTION FRAMEWORK - SETUP VALIDATION")
    print("=" * 60 + "\n")
    
    # Check environment
    env_status = check_environment()
    
    results = []
    
    # Test OpenAI if available
    if env_status['openai_available']:
        results.append(("OpenAI Provider", test_openai_provider()))
    else:
        print("\n⚠ Skipping OpenAI tests - package not installed")
    
    # Test Azure if configured
    if env_status['azure_configured'] and env_status['openai_available']:
        results.append(("Azure OpenAI", test_azure_provider()))
    
    # Test Databricks if available
    if env_status['spark_available']:
        results.append(("Databricks Mode", test_databricks_mode()))
    
    # Test extraction pipeline if OpenAI is available
    if env_status['openai_key'] and env_status['openai_available']:
        results.append(("Extraction Pipeline", test_extraction_pipeline()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
    elif passed > 0:
        print("\n⚠ Some tests failed. Check the output above for details.")
    else:
        print("\n❌ No tests passed. Please check your configuration.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)