import pandas as pd
from pathlib import Path


def read_csv(path: str) -> pd.DataFrame:
    """Read CSV with Azure Document Intelligence format."""
    df = pd.read_csv(path)

    # Ensure required columns exist
    required = ["path", "paragraphs", "tables"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse JSON columns if stored as strings
    import json

    for col in ["paragraphs", "tables"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

    return df


def save_results(df: pd.DataFrame, path: str):
    """Save results maintaining structure."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(df, "toPandas"):
        df = df.toPandas()

    df.to_csv(path, index=False)
