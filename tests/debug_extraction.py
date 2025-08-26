#!/usr/bin/env python
"""
Debug extraction to see what's happening
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
from xtract.utils.config_loader import load_config, load_yaml
from xtract.models.providers import ModelProvider
from xtract.core.chunker import Chunker

print("=" * 50)
print("Extraction Debug")
print("=" * 50)

# Load one row of data
df = read_csv("data/documents.csv")
test_row = df.iloc[0]

print(f"\nTesting with document: {test_row['path']}")

# Load chunker
config = load_config("config/settings.yaml")
chunker = Chunker("config/settings.yaml")

# Chunk the content
paragraphs = test_row['paragraphs']
tables = test_row.get('tables')
chunks = chunker.chunk(paragraphs, tables)

print(f"\nCreated {len(chunks)} chunks")
if chunks:
    print(f"First chunk preview: {chunks[0][:200]}...")

# Load an extraction config
contract_config = load_yaml("config/extraction/contract_type.yaml")
print(f"\nContract type extraction config:")
print(f"  Prompt template length: {len(contract_config.get('prompt', ''))}")

# Format the prompt
prompt_template = contract_config.get('prompt', '')
prompt = prompt_template.format(content=chunks[0] if chunks else "")
print(f"\nFormatted prompt preview:")
print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

# Test model inference
model = ModelProvider(config.get("model"))
print(f"\nTesting model inference...")
try:
    response = model.infer(prompt)
    print(f"\nModel response:")
    print(response)
    
    # Try to parse as JSON
    try:
        parsed = json.loads(response)
        print(f"\n✓ Successfully parsed as JSON:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        print(f"\n✗ Failed to parse as JSON: {e}")
        print("Raw response type:", type(response))
        
    # Test if response contains expected fields
    if isinstance(response, str):
        if "text" in response.lower():
            print("✓ Response contains 'text'")
        if "value" in response.lower():
            print("✓ Response contains 'value'")
        if "confidence" in response.lower():
            print("✓ Response contains 'confidence'")
            
except Exception as e:
    print(f"\n✗ Model inference failed: {e}")
    import traceback
    traceback.print_exc()

# Test the matching logic
print("\n" + "=" * 50)
print("Testing Matching Logic")
print("=" * 50)

from difflib import SequenceMatcher

# Simulate an extraction
test_extraction = {
    "text": "MASTER SERVICE AGREEMENT",
    "value": "MSA",
    "confidence": 0.95
}

print(f"\nTest extraction: {test_extraction}")
print(f"Chunk contains text: {'MASTER SERVICE AGREEMENT' in chunks[0] if chunks else 'N/A'}")

# Test exact match
if test_extraction.get("text") in chunks[0]:
    print("✓ Exact match would succeed")
else:
    print("✗ Exact match would fail")
    
    # Test fuzzy match
    ratio = SequenceMatcher(None, test_extraction.get("text", ""), chunks[0]).ratio()
    print(f"Fuzzy match ratio: {ratio:.2f}")
    if ratio >= 0.85:
        print("✓ Would match at threshold 0.85")
    elif ratio >= 0.75:
        print("✓ Would match at threshold 0.75")
    elif ratio >= 0.65:
        print("✓ Would match at threshold 0.65")
    else:
        print("✗ Would not match at any threshold")

print("\n" + "=" * 50)
print("Debug complete")
print("=" * 50)