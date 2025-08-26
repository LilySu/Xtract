#!/usr/bin/env python
"""
Test evaluation integration
"""

import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from xtract.utils.io import read_csv
from xtract.factory.entity_factory import EntityFactory

print("=" * 50)
print("Evaluation Integration Test")
print("=" * 50)

# Load entity factory
factory = EntityFactory(
    extraction_dir=str(Path(__file__).parent.parent / "config" / "extraction"),
    evaluation_dir=str(Path(__file__).parent.parent / "config" / "evaluation"),
)

# Check what evaluations are loaded
print("\nLoaded evaluations:")
for eval_name, eval_config in factory.evaluations.items():
    print(f"  - {eval_name}: {eval_config.get('method', 'unknown')}")

# Check entity configurations
print("\nEntity configurations:")
for entity in factory.get_entities():
    print(f"\n{entity.name}:")
    print(f"  Model: {entity.model_override or 'default'}")
    print("  Active evaluations:")
    active_evals = entity.get_active_evaluations()
    for eval_name, eval_config in active_evals.items():
        print(f"    - {eval_name}: {eval_config.get('method')}")

# Test a single extraction with evaluation
print("\n" + "=" * 50)
print("Testing extraction with evaluations")
print("=" * 50)

df = read_csv(str(Path(__file__).parent.parent / "data" / "documents.csv"))
test_row = df.iloc[0]

from xtract.core.extractor import Extractor

extractor = Extractor(str(Path(__file__).parent.parent / "config" / "settings.yaml"))

# Process just one row
import pandas as pd

test_df = pd.DataFrame([test_row])

# Extract
result_df, extraction_results = extractor.extract(test_df)

# Check columns created
print("\nColumns created:")
for col in result_df.columns:
    if col not in ["path", "paragraphs", "tables"]:
        print(f"  - {col}")
        # Show the value
        value = result_df[col].iloc[0]
        if isinstance(value, list) and value:
            print(
                f"    Value: {value[0] if len(value) == 1 else f'{len(value)} items'}"
            )
        else:
            print(f"    Value: {value}")

# Check extraction results
print("\n" + "=" * 50)
print("Extraction Results")
print("=" * 50)

for entity_name, results in extraction_results.items():
    print(f"\n{entity_name}:")
    if results and results[0]:
        if isinstance(results[0], list) and results[0]:
            extracted = results[0][0]
            print(f"  Extracted: {json.dumps(extracted, indent=2)}")

            # Check confidence field
            if "confidence" in extracted:
                conf = extracted["confidence"]
                if conf in ["high", "medium", "low"]:
                    print(f"  ✓ Confidence is categorical: {conf}")
                else:
                    print(f"  ✗ Confidence should be categorical, got: {conf}")
        else:
            print("  No extraction")

print("\n" + "=" * 50)
print("Test complete")
print("=" * 50)
