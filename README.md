# 🤖 MLOps Batch Job — Rolling-Mean Signal Pipeline

A minimal, reproducible MLOps-style batch job that computes a binary trading signal from OHLCV data using a configurable rolling mean. Fully Dockerized with structured logging and machine-readable metrics output.

---

## Project Structure

```
mlops_task/
├── run.py           # Main pipeline script
├── config.yaml      # Seed, window, version config
├── data.csv         # 10,000-row OHLCV dataset
├── Dockerfile       # One-command Docker build + run
├── requirements.txt
├── README.md
├── metrics.json     # Sample output (successful run)
└── run.log          # Sample log (successful run)
```

---

## How It Works

1. **Load config** (`config.yaml`) — validate `seed`, `window`, `version`
2. **Set seed** — `numpy.random.seed(seed)` for deterministic runs
3. **Load dataset** (`data.csv`) — validate non-empty, `close` column present
4. **Rolling mean** — `close.rolling(window=window)` (first `window-1` rows excluded from signal)
5. **Signal** — `1` if `close > rolling_mean`, else `0`
6. **Write** `metrics.json` (always) and `run.log`

---

## Local Run

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

---

## Docker Build & Run

```bash
# Build
docker build -t mlops-task .

# Run (prints metrics JSON to stdout, exits 0 on success)
docker run --rm mlops-task
```

To extract output files from the container:

```bash
docker run --rm -v "$(pwd)/output:/app/output" \
  mlops-task \
  python run.py \
    --input data.csv \
    --config config.yaml \
    --output output/metrics.json \
    --log-file output/run.log
```

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 24,
  "seed": 42,
  "status": "success"
}
```

> `rows_processed` is 9996 (not 10000) because the first `window-1 = 4` rows lack a full rolling window and are excluded from signal computation. This is documented behaviour.

---

## Error Output Example

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Dataset error: Required column 'close' not found. Columns present: ['open', 'high']"
}
```

---

## Config Reference

| Key | Type | Description |
|---|---|---|
| `seed` | int | NumPy random seed for reproducibility |
| `window` | int | Rolling mean window size (≥ 1) |
| `version` | str | Pipeline version tag (included in metrics output) |

---

## Validation & Error Handling

| Case | Behaviour |
|---|---|
| Missing config file | Error metrics JSON + exit 1 |
| Missing config key | Error metrics JSON + exit 1 |
| Missing input CSV | Error metrics JSON + exit 1 |
| Empty CSV | Error metrics JSON + exit 1 |
| Missing `close` column | Error metrics JSON + exit 1 |
| Non-numeric `close` | Error metrics JSON + exit 1 |
| Successful run | Full metrics JSON + exit 0 |

---

## Reproducibility

Running with the same `config.yaml` always produces identical `metrics.json` output. The seed is set before any processing via `numpy.random.seed(seed)`.

```bash
python run.py --input data.csv --config config.yaml --output m1.json --log-file l1.log
python run.py --input data.csv --config config.yaml --output m2.json --log-file l2.log
diff m1.json m2.json   # → no differences (except latency_ms)
```

---

## Requirements

- Python 3.9+
- numpy ≥ 1.23
- pandas ≥ 1.5
- pyyaml ≥ 6.0
