"""
run.py — MLOps batch job: rolling-mean signal pipeline.

Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: str) -> logging.Logger:
    """Configure root logger with both file and console handlers."""
    logger = logging.getLogger("mlops")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler — DEBUG and above
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Metrics writer
# ──────────────────────────────────────────────────────────────────────────────

def write_metrics(path: str, payload: dict[str, Any]) -> None:
    """Write metrics dict as pretty JSON. Always called — success and error."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Config loading & validation
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str) -> dict:
    """
    Parse YAML config and validate required fields.

    Returns validated config dict.
    Raises ValueError on missing keys or wrong types.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping (key: value pairs).")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    if not isinstance(cfg["seed"], int):
        raise ValueError(f"Config 'seed' must be an integer, got: {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError("Config 'version' must be a non-empty string.")

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading & validation
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(input_path: str) -> pd.DataFrame:
    """
    Read CSV and validate it is usable.

    Returns a DataFrame with at least a 'close' column.
    Raises appropriate errors for all bad-input cases.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty (no rows).")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Columns present: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df["close"]):
        raise ValueError("Column 'close' must contain numeric values.")

    if df["close"].isna().all():
        raise ValueError("Column 'close' contains only NaN values.")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────────────────────────────────────

def compute_rolling_mean(close: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling mean of close prices.

    The first (window-1) rows will be NaN; these rows are excluded
    from signal computation to keep the signal well-defined.
    """
    return close.rolling(window=window, min_periods=window).mean()


def compute_signal(close: pd.Series, rolling_mean: pd.Series) -> pd.Series:
    """
    Binary signal: 1 if close > rolling_mean, else 0.
    Rows where rolling_mean is NaN (warm-up period) are excluded (set to NaN).
    """
    signal = pd.Series(np.nan, index=close.index)
    valid = rolling_mean.notna()
    signal[valid] = (close[valid] > rolling_mean[valid]).astype(int)
    return signal


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    config_path: str,
    output_path: str,
    log_file: str,
) -> int:
    """
    Execute the full batch pipeline.

    Returns exit code: 0 = success, 1 = failure.
    """
    logger = setup_logging(log_file)
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("Job started")
    logger.info("  input    : %s", input_path)
    logger.info("  config   : %s", config_path)
    logger.info("  output   : %s", output_path)
    logger.info("  log_file : %s", log_file)
    logger.info("=" * 60)

    # Default version for error payloads before config is loaded
    version = "unknown"

    # ── Step 1: Load & validate config ────────────────────────────────────
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        logger.error("Config error: %s", exc)
        write_metrics(output_path, {
            "version": version,
            "status": "error",
            "error_message": f"Config error: {exc}",
        })
        return 1

    version = cfg["version"]
    seed = cfg["seed"]
    window = cfg["window"]

    logger.info("Config loaded and validated")
    logger.info("  version : %s", version)
    logger.info("  seed    : %s", seed)
    logger.info("  window  : %s", window)

    # Set seed for reproducibility
    np.random.seed(seed)
    logger.debug("NumPy random seed set to %d", seed)

    # ── Step 2: Load & validate dataset ───────────────────────────────────
    try:
        df = load_dataset(input_path)
    except Exception as exc:
        logger.error("Dataset error: %s", exc)
        write_metrics(output_path, {
            "version": version,
            "status": "error",
            "error_message": f"Dataset error: {exc}",
        })
        return 1

    logger.info("Dataset loaded: %d rows, %d columns", len(df), len(df.columns))
    logger.debug("Columns: %s", list(df.columns))
    logger.info("close — min=%.4f  max=%.4f  nulls=%d",
                df["close"].min(), df["close"].max(), df["close"].isna().sum())

    # ── Step 3: Rolling mean ───────────────────────────────────────────────
    logger.info("Computing rolling mean (window=%d) ...", window)
    rolling_mean = compute_rolling_mean(df["close"], window)
    warm_up_rows = window - 1
    valid_rows = rolling_mean.notna().sum()
    logger.info("Rolling mean computed — warm-up rows excluded: %d  valid rows: %d",
                warm_up_rows, valid_rows)

    # ── Step 4: Signal ─────────────────────────────────────────────────────
    logger.info("Generating binary signal (close > rolling_mean → 1 else 0) ...")
    signal = compute_signal(df["close"], rolling_mean)
    valid_signal = signal.dropna()
    rows_processed = len(valid_signal)
    signal_rate = float(valid_signal.mean())
    logger.info("Signal generated — rows used: %d  signal_rate: %.6f",
                rows_processed, signal_rate)
    logger.debug("Signal value counts: %s", valid_signal.value_counts().to_dict())

    # ── Step 5: Metrics + timing ───────────────────────────────────────────
    latency_ms = int((time.time() - t_start) * 1000)

    metrics = {
        "version": version,
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": latency_ms,
        "seed": seed,
        "status": "success",
    }

    write_metrics(output_path, metrics)
    logger.info("Metrics written to %s", output_path)
    logger.info("Metrics summary: %s", json.dumps(metrics))

    logger.info("=" * 60)
    logger.info("Job completed successfully | latency=%d ms", latency_ms)
    logger.info("=" * 60)

    # Print final metrics to stdout (required by Docker spec)
    print(json.dumps(metrics, indent=2))

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps batch job: rolling-mean signal pipeline",
    )
    parser.add_argument("--input",    required=True, help="Path to input CSV (OHLCV)")
    parser.add_argument("--config",   required=True, help="Path to config YAML")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for log output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = run_pipeline(
        input_path=args.input,
        config_path=args.config,
        output_path=args.output,
        log_file=args.log_file,
    )
    sys.exit(exit_code)
