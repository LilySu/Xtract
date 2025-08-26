# Xtract

A flexible extraction and evaluation framework for document processing.

## Installation

```bash
pip install -e .
```

## Usage

```python
from xtract.core import Extractor

# Initialize extractor
extractor = Extractor(config_path="config/settings.yaml")

# Extract entities from document
results = extractor.extract(document_path="path/to/document.pdf")
```

## Configuration

Configuration files are located in the `config/` directory:
- `settings.yaml`: Main configuration settings
- `extraction/`: Entity extraction configurations
- `evaluation/`: Evaluation metrics configurations

## Examples

See the `examples/` directory for usage examples.


```
python test_extractor_optimizations.py -v
```