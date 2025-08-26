#!/usr/bin/env python
"""
Debug why signing party extraction is empty
"""
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

from xtract.utils.io import read_csv
from xtract.utils.config_loader import load_yaml
from xtract.models.providers import ModelProvider
from xtract.core.chunker import Chunker

print("=" * 50)
print("Signing Party Debug")
print("=" * 50)

# Load data and config
df = read_csv("data/documents.csv")
test_row = df.iloc[0]
config = load_yaml("config/settings.yaml")

# Chunk the content
chunker = Chunker("config/settings.yaml")
chunks = chunker.chunk(test_row['paragraphs'], test_row.get('tables'))

print(f"\nChunk content:")
print(chunks[0][:500] + "...")

# Load signing party config
signing_config = load_yaml("config/extraction/signing_party.yaml")
prompt_template = signing_config.get('prompt', '')
prompt = prompt_template.format(content=chunks[0])

print(f"\nPrompt being sent:")
print(prompt[:600] + "...")

# Test model inference
model = ModelProvider(config.get("model"))
print(f"\nCalling model...")
response = model.infer(prompt, model_name="gpt-4o-mini")

print(f"\nRaw response:")
print(response)

# Parse response
if "```" in response:
    response = response.replace("```json", "").replace("```", "")
    start = response.find("{")
    end = response.rfind("}") + 1
    if start != -1 and end > start:
        response = response[start:end]

response = response.strip()

try:
    parsed = json.loads(response)
    print(f"\nParsed response:")
    print(json.dumps(parsed, indent=2))
    
    # Check if text field exists
    if "text" in parsed:
        print(f"\n✓ Text field: {parsed['text']}")
    elif "parties" in parsed:
        # Create text field from parties
        party_names = [p.get("party_name", "") for p in parsed["parties"]]
        text = ", ".join(party_names)
        print(f"\n✓ Created text from parties: {text}")
        
    # Check matching
    if "text" in parsed or "parties" in parsed:
        text_to_match = parsed.get("text", "") or ", ".join([p.get("party_name", "") for p in parsed.get("parties", [])])
        print(f"\nChecking if '{text_to_match}' is in chunk...")
        if text_to_match in chunks[0]:
            print("✓ Would match exactly")
        else:
            print("✗ No exact match")
            # Check what parties are actually in the text
            import re
            print("\nLooking for company names in chunk:")
            companies = re.findall(r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*(?:,? (?:Inc\.|LLC|Corp\.|Corporation|Ltd\.|Company))?)", chunks[0])
            for company in companies[:5]:
                print(f"  - Found: {company}")
                
except json.JSONDecodeError as e:
    print(f"\n✗ Failed to parse JSON: {e}")

print("\n" + "=" * 50)
print("Debug complete")
print("=" * 50)