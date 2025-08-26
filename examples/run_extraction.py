#!/usr/bin/env python
import sys
import json
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from xtract.core.extractor import Extractor
from xtract.core.evaluator import Evaluator
from xtract.utils.io import read_csv, save_results
from xtract.utils.config_loader import load_config


def main():
    # Load configuration
    config = load_config("config/settings.yaml")
    print(f"Running in {config['mode']} mode")

    if config["mode"] == "databricks":
        run_databricks(config)
    else:
        run_local(config)


def run_local(config):
    """Run extraction locally with pandas"""

    # Read data
    print("Loading data...")
    df = read_csv("data/documents.csv")
    print(f"Loaded {len(df)} documents")

    # Sample if configured
    sample_size = config.get("sample_size")
    if sample_size:
        print(f"Sampling {sample_size} rows")
        df = df.head(sample_size)

    # Extract entities
    print("Extracting entities...")
    extractor = Extractor(config_path="config/settings.yaml")
    df_extracted, extraction_results = extractor.extract(df)

    # Print extraction summary
    print("\nExtraction Summary:")
    for entity_name, results in extraction_results.items():
        total = sum(len(r) if isinstance(r, list) else 1 for r in results if r)
        print(f"  {entity_name}: {total} extractions")

    # Run additional evaluations if needed
    if config.get("evaluation", {}).get("enabled", True):
        print("\nRunning evaluations...")
        evaluator = Evaluator()
        df_final = evaluator.evaluate(df_extracted, extraction_results)
    else:
        df_final = df_extracted

    # Save results
    print("\nSaving results...")
    save_results(df_final, "output/results.csv")

    # Save extraction summary
    summary = {
        "config": {
            "mode": config["mode"],
            "sample_size": config.get("sample_size"),
            "chunking_strategy": config.get("chunking", {}).get("strategy"),
            "matching": config.get("extraction", {}).get("matching", {}),
        },
        "results": {},
    }

    for col in df_final.columns:
        if col.endswith("_extracted"):
            entity_name = col.replace("_extracted", "")
            extractions = df_final[col].tolist()
            summary["results"][entity_name] = {
                "total_extractions": sum(
                    len(e) if isinstance(e, list) else 0 for e in extractions
                ),
                "documents_with_extraction": sum(
                    1 for e in extractions if e and len(e) > 0
                ),
                "total_documents": len(df_final),
            }

            # Add evaluation results
            eval_cols = [c for c in df_final.columns if c.startswith(f"{entity_name}_")]
            if eval_cols:
                summary["results"][entity_name]["evaluations"] = {}
                for eval_col in eval_cols:
                    if not eval_col.endswith("_extracted"):
                        eval_name = eval_col.replace(f"{entity_name}_", "")
                        values = df_final[eval_col].value_counts().to_dict()
                        summary["results"][entity_name]["evaluations"][eval_name] = (
                            values
                        )

    with open("output/extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done! Results saved to output/")
    print("  - Full results: output/results.csv")
    print("  - Summary: output/extraction_summary.json")


def run_databricks(config):
    """Run extraction on Databricks with Spark"""
    from pyspark.sql import SparkSession

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("xtract-extraction")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )

    print("Loading data...")
    df = spark.read.option("multiLine", True).json("data/documents.json")
    # Or read parquet: df = spark.read.parquet("data/documents.parquet")

    print(f"Loaded {df.count()} documents")

    # Sample if configured
    sample_size = config.get("sample_size")
    if sample_size:
        print(f"Sampling {sample_size} rows")
        df = df.limit(sample_size)

    # Extract entities
    print("Extracting entities...")
    extractor = Extractor(config_path="config/settings.yaml")
    df_extracted, extraction_results = extractor.extract(df)

    # Cache for performance
    df_extracted.cache()

    # Print extraction summary
    print("\nExtraction Summary:")
    for entity_name in extraction_results.keys():
        col_name = f"{entity_name}_extracted"
        count = df_extracted.filter(
            f"{col_name} is not null and {col_name} != '[]'"
        ).count()
        print(f"  {entity_name}: {count} documents with extractions")

    # Save results
    print("\nSaving results...")
    output_path = "output/spark_results"
    df_extracted.coalesce(1).write.mode("overwrite").parquet(output_path)

    # Save summary
    summary = {
        "total_documents": df_extracted.count(),
        "entities_extracted": list(extraction_results.keys()),
        "output_path": output_path,
    }

    with open("output/spark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done! Results saved to {output_path}")
    spark.stop()


if __name__ == "__main__":
    main()
