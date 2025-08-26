import pandas as pd
from pathlib import Path
import json
from typing import Union, Any

# Try to import Spark components
try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, from_json
    from pyspark.sql.types import StructType, StructField, StringType
    SPARK_AVAILABLE = True
except ImportError:
    SparkDataFrame = None
    SPARK_AVAILABLE = False


def read_csv(path: str, mode: str = "local") -> Union[pd.DataFrame, Any]:
    """
    Read CSV with Azure Document Intelligence format.
    
    Args:
        path: Path to CSV file
        mode: "local" for pandas, "databricks" for Spark DataFrame
    
    Returns:
        DataFrame (pandas or Spark depending on mode)
    """
    if mode == "databricks" and SPARK_AVAILABLE:
        return read_csv_spark(path)
    else:
        return read_csv_pandas(path)


def read_csv_pandas(path: str) -> pd.DataFrame:
    """Read CSV using pandas."""
    df = pd.read_csv(path)

    # Ensure required columns exist
    required = ["path", "paragraphs"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse JSON columns if stored as strings
    for col in ["paragraphs", "tables"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
            )

    # Ensure tables column exists even if empty
    if "tables" not in df.columns:
        df["tables"] = [[] for _ in range(len(df))]

    return df


def read_csv_spark(path: str) -> Any:
    """Read CSV using Spark."""
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark not available. Install with: pip install pyspark")
    
    # Get Spark session
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("DocumentExtractor").getOrCreate()
    
    # Read CSV
    df = spark.read.csv(path, header=True, inferSchema=False)
    
    # Ensure required columns exist
    required = ["path", "paragraphs"]
    missing = [col_name for col_name in required if col_name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add tables column if it doesn't exist
    if "tables" not in df.columns:
        df = df.withColumn("tables", col("paragraphs").cast("string").substr(0, 0))  # Empty string
    
    # Note: In Spark, we keep JSON columns as strings and parse them in the extraction
    # This is more efficient for large datasets
    
    return df


def save_results(df: Union[pd.DataFrame, Any], path: str, mode: str = "local"):
    """
    Save results maintaining structure.
    
    Args:
        df: DataFrame to save (pandas or Spark)
        path: Output path
        mode: "local" or "databricks"
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "databricks" and SPARK_AVAILABLE and hasattr(df, 'toPandas'):
        # Handle Spark DataFrame
        save_results_spark(df, str(output_path))
    else:
        # Handle pandas DataFrame
        if hasattr(df, "toPandas"):
            # Convert Spark to pandas if needed
            df = df.toPandas()
        save_results_pandas(df, str(output_path))


def save_results_pandas(df: pd.DataFrame, path: str):
    """Save pandas DataFrame."""
    # Convert complex objects to JSON strings for CSV storage
    for col in df.columns:
        if df[col].dtype == "object":
            # Check if column contains complex objects
            first_val = df[col].iloc[0] if len(df) > 0 else None
            if isinstance(first_val, (list, dict)):
                df[col] = df[col].apply(lambda x: json.dumps(x) if x else "")
    
    df.to_csv(path, index=False)


def save_results_spark(df: Any, path: str):
    """Save Spark DataFrame."""
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark not available")
    
    # For Spark, we have options:
    # 1. Save as CSV (single file or partitioned)
    # 2. Save as Parquet (more efficient for large data)
    # 3. Convert to pandas and save
    
    # Option 1: Save as single CSV file (good for small-medium datasets)
    if path.endswith('.csv'):
        # Coalesce to single partition for single file output
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path.replace('.csv', '_spark'))
        
        # Move the single part file to the desired location
        import glob
        import shutil
        spark_files = glob.glob(f"{path.replace('.csv', '_spark')}/part-*.csv")
        if spark_files:
            shutil.move(spark_files[0], path)
            # Clean up the spark directory
            shutil.rmtree(path.replace('.csv', '_spark'))
    
    # Option 2: Save as Parquet (better for large datasets)
    elif path.endswith('.parquet'):
        df.write.mode("overwrite").parquet(path)
    
    # Option 3: Default - save as partitioned CSV
    else:
        df.write.mode("overwrite").option("header", "true").csv(path)


# Utility functions for Databricks environments
def is_databricks_environment() -> bool:
    """Check if running in Databricks environment."""
    try:
        # Databricks-specific environment variable
        import os
        return "DATABRICKS_RUNTIME_VERSION" in os.environ
    except:
        return False


def get_dbfs_path(path: str) -> str:
    """Convert local path to DBFS path if in Databricks."""
    if is_databricks_environment() and not path.startswith("/dbfs"):
        return f"/dbfs/{path.lstrip('/')}"
    return path


# Export utility functions for working with different data formats
def convert_to_pandas(df: Any) -> pd.DataFrame:
    """Convert any DataFrame type to pandas."""
    if isinstance(df, pd.DataFrame):
        return df
    elif hasattr(df, 'toPandas'):
        return df.toPandas()
    else:
        raise TypeError(f"Cannot convert {type(df)} to pandas DataFrame")


def convert_to_spark(df: pd.DataFrame) -> Any:
    """Convert pandas DataFrame to Spark."""
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark not available")
    
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("DocumentExtractor").getOrCreate()
    
    # Convert complex columns to JSON strings first
    for col in df.columns:
        if df[col].dtype == "object":
            first_val = df[col].iloc[0] if len(df) > 0 else None
            if isinstance(first_val, (list, dict)):
                df[col] = df[col].apply(lambda x: json.dumps(x) if x else "")
    
    return spark.createDataFrame(df)