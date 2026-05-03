"""Data loading, cleaning, and temporal train/test splitting."""

import os
import pandas as pd
import numpy as np


def load_raw(data_dir: str, filename: str = None) -> pd.DataFrame:
    """Load the raw Lending Club data file from data_dir.

    If filename is not provided, finds the first .gzip/.gz/.csv file in data_dir.
    Auto-detects compression from file magic bytes.
    """
    if filename is None:
        for ext in (".gzip", ".gz", ".csv"):
            matches = [f for f in os.listdir(data_dir) if f.endswith(ext)]
            if matches:
                filename = matches[0]
                break
        if filename is None:
            raise FileNotFoundError(f"No data files found in {data_dir}")
    path = os.path.join(data_dir, filename)
    # Auto-detect: file may have .gzip extension but be plain CSV
    compression = "infer"
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic != b"\x1f\x8b":  # not actual gzip
        compression = None
    return pd.read_csv(path, compression=compression, low_memory=False)


def define_target(df: pd.DataFrame, col: str = "loan_status") -> pd.DataFrame:
    """Filter to 'Fully Paid' and 'Charged Off', create binary target."""
    df = df[df[col].isin(["Fully Paid", "Charged Off"])].copy()
    df["target"] = (df[col] == "Charged Off").astype(int)
    return df


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "issue_d",
    train_end: str = "Jul-2015",
    test_end: str = "Dec-2018",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train through train_end, test from next month through test_end."""
    df[date_col] = pd.to_datetime(df[date_col], format="%b-%Y")
    train_cutoff = pd.to_datetime(train_end, format="%b-%Y")
    test_cutoff = pd.to_datetime(test_end, format="%b-%Y")
    train = df[df[date_col] <= train_cutoff].copy()
    test = df[(df[date_col] > train_cutoff) & (df[date_col] <= test_cutoff)].copy()
    return train, test
