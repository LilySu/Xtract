#!/usr/bin/env python
"""
Quick test script to verify the CSV data loads correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from xtract.utils.io import read_csv
from xtract.utils.config_loader import load_config


def test_csv_load():
    """Test loading the sample CSV using xtract utilities"""
    print("Testing CSV load with xtract.utils.io...")

    # Get project root path
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "documents.csv"

    # Load CSV using the existing utility
    df = read_csv(str(csv_path))
    print(f"✓ Loaded {len(df)} rows")

    # Validate data structure
    for idx, row in df.iterrows():
        print(f"\nRow {idx + 1}: {row['path']}")

        # Check paragraphs
        paragraphs = row["paragraphs"]
        assert isinstance(paragraphs, list), (
            f"Paragraphs should be list, got {type(paragraphs)}"
        )
        print(f"  - Paragraphs: {len(paragraphs)} items")

        if paragraphs:
            first_para = paragraphs[0]
            assert "text" in first_para, "Paragraph missing 'text' field"
            assert "page_number" in first_para, "Paragraph missing 'page_number' field"
            print(f"    First text: '{first_para['text'][:50]}...'")

        # Check tables
        tables = row["tables"]
        assert isinstance(tables, list), f"Tables should be list, got {type(tables)}"
        print(f"  - Tables: {len(tables)} items")

        if tables and len(tables) > 0:
            first_table = tables[0]
            assert "text" in first_table, "Table missing 'text' field"
            print("    Table data present")

    print("\n✅ All tests passed! CSV is properly formatted.")
    return df


def test_config_load():
    """Test loading configuration"""
    print("\nTesting config load...")

    # Get project root path
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "settings.yaml"

    if config_path.exists():
        config = load_config(str(config_path))
        print("✓ Loaded config")
        print(f"  - Mode: {config.get('mode', 'not set')}")
        print(f"  - Sample size: {config.get('sample_size', 'not set')}")
        print(
            f"  - Chunking strategy: {config.get('chunking', {}).get('strategy', 'not set')}"
        )
    else:
        print("⚠ Config file not found at config/settings.yaml")
        print("  Using defaults from code")


def test_extraction_preview(df):
    """Preview what will be extracted"""
    print("\n" + "=" * 50)
    print("Preview of extraction targets:")
    print("=" * 50)

    for idx, row in df.iterrows():
        print(f"\nDocument: {row['path']}")

        # Combine all text
        all_text = " ".join([p["text"] for p in row["paragraphs"]])

        # Look for contract type indicators
        contract_types = [
            "MASTER SERVICE AGREEMENT",
            "NON-DISCLOSURE AGREEMENT",
            "STATEMENT OF WORK",
            "MSA",
            "NDA",
            "SOW",
        ]
        found_type = None
        for ct in contract_types:
            if ct in all_text.upper():
                found_type = ct
                break
        print(f"  Likely contract type: {found_type or 'Unknown'}")

        # Look for parties
        import re

        parties = re.findall(r"between ([^,]+),", all_text, re.IGNORECASE)
        if parties:
            print(f"  Potential parties: {parties[0][:50]}...")

        # Check for dates
        dates = re.findall(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
            all_text,
        )
        if dates:
            print(f"  Found dates: {dates[0]}")


if __name__ == "__main__":
    print("Xtract Sample Data Test")
    print("-" * 50)

    try:
        df = test_csv_load()
        test_config_load()
        test_extraction_preview(df)

        print("\n" + "=" * 50)
        print("✅ Sample data is ready for extraction!")
        print("\nNext steps:")
        print("1. Update .env with your API keys")
        print("2. Run: python examples/run_extraction.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. The CSV file is in data/documents.csv")
        print("2. The test is being run from the tests directory")
        import traceback

        traceback.print_exc()
