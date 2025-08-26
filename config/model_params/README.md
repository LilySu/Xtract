# Configuration Guide

## Overview

The Document Intelligence Extraction Framework supports multiple modes and providers for maximum flexibility. By default, it uses **local mode** with **OpenAI**, but can be configured for enterprise environments using **Databricks** and **Azure OpenAI**.

## Quick Start (Default Configuration)

The default configuration works out of the box with minimal setup:

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. Use the default `config/settings.yaml`:
```yaml
mode: local  # Uses pandas for data processing
model:
  provider: openai
  model_name: gpt-4o-mini
```

That's it! The system will use the simple OpenAI provider without requiring additional configuration files.

## Mode Configuration

### 1. Local Mode (Default)
Uses pandas DataFrames for data processing. Best for:
- Development and testing
- Small to medium datasets (< 1GB)
- Single machine processing

**Configuration:**
```yaml
mode: local
```

### 2. Databricks/Spark Mode
Uses PySpark for distributed processing. Best for:
- Large datasets (> 1GB)
- Production environments
- Distributed processing across clusters

**Prerequisites:**
```bash
pip install pyspark
```

**Configuration:**
```yaml
mode: databricks

spark:
  use_partitions: true
  num_partitions: 200  # Adjust based on cluster size
```

**Important Notes:**
- If PySpark is not installed, the system automatically falls back to local mode
- In Databricks notebooks, data is automatically handled as Spark DataFrames
- For local Spark testing, data is converted between pandas and Spark as needed

## Model Provider Configuration

### Option 1: Simple Mode (Default)
Uses OpenAI directly without additional configuration files.

**Configuration in `settings.yaml`:**
```yaml
model:
  provider: openai
  model_name: gpt-4o-mini
  temperature: 0.0
  max_tokens: 4096
```

### Option 2: Advanced Mode with Routing
For multiple providers or Azure OpenAI, create a `config/model_params/` directory with provider configurations.

#### Setting up Azure OpenAI

1. **Create the configuration directory:**
```bash
mkdir -p config/model_params
```

2. **Create `config/model_params/model_configs.yaml`:**
```yaml
routing:
  default_provider: azure_openai  # Change default to Azure
  patterns:
    gpt: openai
    azure: azure_openai
    o1: openai
    o3: openai

providers:
  openai:
    class: OpenAIProvider
    config_file: openai.yaml
  azure_openai:
    class: AzureOpenAIProvider
    config_file: azure_openai.yaml
```

3. **Create `config/model_params/azure_openai.yaml`:**
```yaml
endpoint: ${AZURE_OPENAI_ENDPOINT}  # https://your-resource.openai.azure.com/
api_key: ${AZURE_OPENAI_API_KEY}
api_version: "2024-10-21"

models:
  gpt-4o-mini:
    deployment: your-gpt-4o-mini-deployment-name
    temperature: 0.0
    max_tokens: 4096
  gpt-4o:
    deployment: your-gpt-4o-deployment-name
    temperature: 0.0
    max_tokens: 4096
```

4. **Set environment variables:**
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-azure-api-key"
```

5. **Update `settings.yaml`:**
```yaml
model:
  provider: azure_openai  # Switch to Azure
  model_name: gpt-4o-mini
```

#### Using Multiple Providers

You can use different providers for different models by configuring patterns:

```yaml
# In model_configs.yaml
routing:
  patterns:
    gpt: openai          # All GPT models use OpenAI
    azure-gpt: azure_openai  # Models prefixed with 'azure-' use Azure
    claude: anthropic    # Future support
```

Then specify the model name to automatically route to the right provider:
```python
# Automatically uses OpenAI
extractor.model.infer(prompt, model_name="gpt-4o")

# Automatically uses Azure
extractor.model.infer(prompt, model_name="azure-gpt-4o")
```

## Common Configurations

### Development Environment
```yaml
mode: local
sample_size: 10  # Process only 10 documents for testing

model:
  provider: openai
  model_name: gpt-4o-mini
  temperature: 0.0
```

### Production with Databricks and Azure
```yaml
mode: databricks
sample_size: null  # Process all documents

spark:
  use_partitions: true
  num_partitions: 200

model:
  provider: azure_openai
  model_name: gpt-4o
  temperature: 0.0
  
  batching:
    enabled: true
    threshold: 10
    batch_size: 20
```

### High-Performance Local Processing
```yaml
mode: local
sample_size: null

model:
  provider: openai
  model_name: gpt-4o-mini
  
  batching:
    enabled: true
    threshold: 5
    batch_size: 10

extraction:
  matching:
    exact_only: false
    fuzzy_passes: 3
    fuzzy_threshold_1: 0.90
    fuzzy_threshold_2: 0.70
    fuzzy_threshold_3: 0.50
```

## Testing Your Configuration

### Test Databricks Mode
```bash
python -m pytest tests/test_databricks_mode.py -v
```

### Test Azure OpenAI Mode
```bash
python -m pytest tests/test_azure_openai_mode.py -v
```

### Quick Validation Script
```python
from src.extraction.extractor import Extractor
from src.models.providers import ModelProvider

# Test model provider
config = {"provider": "azure_openai", "model_name": "gpt-4o"}  # or "openai"
model = ModelProvider(config)
response = model.infer("Hello, this is a test.")
print(f"Model response: {response}")

# Test extractor mode
extractor = Extractor("config/settings.yaml")
print(f"Running in {extractor.mode} mode")
```

## Troubleshooting

### Issue: "PySpark not available but Spark extraction was called"
**Solution:** Install PySpark or switch to local mode:
```bash
pip install pyspark
# OR change mode to 'local' in settings.yaml
```

### Issue: "Azure OpenAI client not initialized"
**Solution:** Check that you have:
1. Created the Azure configuration files
2. Set the environment variables
3. Installed the openai package: `pip install openai`

### Issue: "No OpenAI API key found"
**Solution:** Set the environment variable:
```bash
export OPENAI_API_KEY="your-key"
# OR for Azure
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
```

### Issue: Automatic fallback to local mode
**Cause:** This happens when Databricks mode is configured but PySpark is not installed.
**Solution:** This is intended behavior to prevent crashes. Install PySpark if you need Databricks mode.

## Best Practices

1. **Start Simple:** Begin with local mode and OpenAI for development
2. **Test Incrementally:** Use `sample_size` to test with small datasets first
3. **Monitor Costs:** Use `gpt-4o-mini` for development, `gpt-4o` for production
4. **Scale Gradually:** Move from local to Databricks only when data size requires it
5. **Environment Variables:** Never commit API keys; always use environment variables
6. **Configuration as Code:** Keep different config files for dev/staging/prod environments