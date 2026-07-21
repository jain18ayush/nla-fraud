"""
Phase 1: Download, clean, feature-engineer, and split fraud transaction data.

Supported --dataset values (in order of preference):
  ieee-fraud-detection        Real e-commerce card fraud (IEEE-CIS / Vesta, ~590k rows).
                              Requires accepting competition rules in browser once.
  kartik2112/fraud-detection  Sparkov synthetic card transactions (~1.3M rows).
  ealtman2019/credit-card-transactions  IBM/TabFormer synthetic (~24M rows).
  synthetic                   Local faker+numpy generator; no Kaggle needed.

Output: data/transactions.parquet  (train_split column: train / val / test)
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def load_config() -> dict:
    with open(ROOT / "configs" / "experiment.yaml") as f:
        return yaml.safe_load(f)


# ── Kaggle helpers ────────────────────────────────────────────────────────────

def _kaggle_competition_download(competition: str, dest: Path) -> Path:
    """Download a competition dataset; returns the unzipped directory."""
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr
        if "403" in stderr or "403 Forbidden" in stderr or "You must accept" in stderr:
            print(
                "\n[ERROR] Kaggle returned 403 for ieee-fraud-detection.\n"
                "You need to accept the competition rules once in your browser:\n"
                "  https://www.kaggle.com/competitions/ieee-fraud-detection/rules\n"
                "Then re-run this script.\n"
                "Alternatively, run with --dataset kartik2112/fraud-detection for the\n"
                "fully-public Sparkov dataset (no rules acceptance required).\n",
                file=sys.stderr,
            )
        else:
            print(f"[ERROR] kaggle CLI failed:\n{stderr}", file=sys.stderr)
        sys.exit(1)
    # Unzip
    for zf in dest.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        zf.unlink()
    return dest


def _kaggle_dataset_download(slug: str, dest: Path) -> Path:
    """Download a regular (non-competition) dataset."""
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] kaggle CLI failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return dest


# ── IEEE-CIS (ieee-fraud-detection) ───────────────────────────────────────────

# Features the MLP will consume — only verbalizable columns plus engineered ones.
IEEE_VERBALIZABLE = {
    # raw inputs
    "TransactionAmt":   "transaction amount in USD",
    "ProductCD":        "product code (W/H/C/S/R)",
    "card4":            "card network (visa/mastercard/etc.)",
    "card6":            "card type (debit/credit)",
    "P_emaildomain":    "purchaser email domain",
    "R_emaildomain":    "recipient email domain",
    "addr1":            "billing zip area code",
    "addr2":            "billing country code",
    "dist1":            "distance between billing and transaction location (miles)",
    "DeviceType":       "device type (desktop/mobile)",
    "DeviceInfo":       "device OS/browser string",
    # time-derived
    "hour_of_day":      "hour of day (0–23)",
    "day_of_week":      "day of week (0=Mon)",
    # engineered behavioral
    "amt_zscore":       "transaction amount z-score vs card's historical mean/std",
    "velocity_1h":      "number of card transactions in the past 1 hour",
    "velocity_24h":     "number of card transactions in the past 24 hours",
    "card_tenure_days": "days since first observed transaction for this card",
}

def _load_ieee(raw_dir: Path, limit: Optional[int]) -> pd.DataFrame:
    tx_file = raw_dir / "train_transaction.csv"
    id_file = raw_dir / "train_identity.csv"
    if not tx_file.exists():
        raise FileNotFoundError(f"Expected {tx_file}; re-run download step.")

    print("Loading IEEE-CIS transaction CSV …")
    tx = pd.read_csv(tx_file, nrows=limit)
    print(f"  {len(tx):,} rows loaded")

    id_df = pd.read_csv(id_file)
    tx = tx.merge(id_df, on="TransactionID", how="left")

    # Time features (TransactionDT is seconds since a reference epoch)
    REF_EPOCH = pd.Timestamp("2017-11-30")
    tx["timestamp"] = REF_EPOCH + pd.to_timedelta(tx["TransactionDT"], unit="s")
    tx["hour_of_day"] = tx["timestamp"].dt.hour
    tx["day_of_week"] = tx["timestamp"].dt.dayofweek

    # Causal behavioral features — computed in chronological order per card
    tx = tx.sort_values("TransactionDT").reset_index(drop=True)
    tx = _add_ieee_behavioral(tx)

    # Keep only verbalizable + label
    keep = list(IEEE_VERBALIZABLE.keys()) + ["TransactionID", "isFraud", "TransactionDT", "timestamp", "card1"]
    tx = tx[[c for c in keep if c in tx.columns]].copy()
    tx = tx.rename(columns={"isFraud": "is_fraud"})
    tx["dataset_source"] = "ieee-fraud-detection"
    return tx


def _add_ieee_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """Compute causal per-card aggregates (no future leakage)."""
    dt_sec = df["TransactionDT"].values
    card_key = df["card1"].values
    amt = df["TransactionAmt"].values

    velocity_1h = np.zeros(len(df), dtype=np.float32)
    velocity_24h = np.zeros(len(df), dtype=np.float32)
    amt_zscore = np.zeros(len(df), dtype=np.float32)
    card_tenure = np.zeros(len(df), dtype=np.float32)

    # Per-card running state
    card_txn_times: dict = {}   # card -> list of timestamps
    card_amt_hist: dict = {}    # card -> list of amounts

    for i in range(len(df)):
        c = card_key[i]
        t = dt_sec[i]
        a = amt[i]

        times = card_txn_times.get(c, [])
        amts = card_amt_hist.get(c, [])

        # Tenure
        card_tenure[i] = (t - times[0]) / 86400.0 if times else 0.0

        # Velocity
        velocity_1h[i] = sum(1 for s in times if t - s <= 3600)
        velocity_24h[i] = sum(1 for s in times if t - s <= 86400)

        # Amount z-score vs history
        if len(amts) >= 2:
            mu = np.mean(amts)
            sigma = np.std(amts) + 1e-6
            amt_zscore[i] = (a - mu) / sigma
        else:
            amt_zscore[i] = 0.0

        times.append(t)
        amts.append(a)
        card_txn_times[c] = times
        card_amt_hist[c] = amts

    df["velocity_1h"] = velocity_1h
    df["velocity_24h"] = velocity_24h
    df["amt_zscore"] = amt_zscore
    df["card_tenure_days"] = card_tenure
    return df


# ── Sparkov (kartik2112/fraud-detection) ──────────────────────────────────────

SPARKOV_VERBALIZABLE = {
    "amt":              "transaction amount in USD",
    "merchant":         "merchant name",
    "category":         "merchant category",
    "gender":           "cardholder gender",
    "city":             "cardholder city",
    "state":            "cardholder state",
    "zip":              "billing zip code",
    "lat":              "cardholder latitude",
    "long":             "cardholder longitude",
    "city_pop":         "population of cardholder city",
    "job":              "cardholder job/occupation",
    "merch_lat":        "merchant latitude",
    "merch_long":       "merchant longitude",
    # time-derived
    "hour_of_day":      "hour of day (0–23)",
    "day_of_week":      "day of week (0=Mon)",
    "age_at_txn":       "cardholder age at time of transaction (years)",
    # engineered
    "amt_zscore":       "transaction amount z-score vs card's historical mean/std",
    "velocity_1h":      "number of card transactions in the past 1 hour",
    "velocity_24h":     "number of card transactions in the past 24 hours",
    "card_tenure_days": "days since first observed transaction for this card",
    "geo_distance_km":  "straight-line distance between cardholder and merchant (km)",
}


def _load_sparkov(raw_dir: Path, limit: Optional[int]) -> pd.DataFrame:
    csvs = sorted(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}")
    print(f"Loading Sparkov CSVs from {raw_dir} …")
    frames = []
    rows = 0
    for p in csvs:
        chunk = pd.read_csv(p)
        frames.append(chunk)
        rows += len(chunk)
        if limit and rows >= limit:
            break
    df = pd.concat(frames, ignore_index=True)
    if limit:
        df = df.iloc[:limit]
    print(f"  {len(df):,} rows loaded")

    # Standardise datetime
    df["timestamp"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Age at transaction
    df["dob"] = pd.to_datetime(df["dob"])
    df["age_at_txn"] = (df["timestamp"] - df["dob"]).dt.days / 365.25

    # Geo distance
    df["geo_distance_km"] = _haversine(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values,
    )

    # Causal behavioral features (use cc_num as card key)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["_ts_sec"] = df["timestamp"].astype(np.int64) // 10**9
    df = _add_sparkov_behavioral(df)

    df = df.rename(columns={"is_fraud": "is_fraud"})  # already correct name
    df["dataset_source"] = "sparkov"

    keep = list(SPARKOV_VERBALIZABLE.keys()) + ["is_fraud", "timestamp", "cc_num", "dataset_source"]
    return df[[c for c in keep if c in df.columns]].copy()


def _add_sparkov_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["_ts_sec"].values
    card_key = df["cc_num"].values
    amt = df["amt"].values

    velocity_1h = np.zeros(len(df), dtype=np.float32)
    velocity_24h = np.zeros(len(df), dtype=np.float32)
    amt_zscore = np.zeros(len(df), dtype=np.float32)
    card_tenure = np.zeros(len(df), dtype=np.float32)

    card_times: dict = {}
    card_amts: dict = {}

    for i in range(len(df)):
        c = card_key[i]
        t = int(ts[i])
        a = float(amt[i])

        times = card_times.get(c, [])
        amts = card_amts.get(c, [])

        card_tenure[i] = (t - times[0]) / 86400.0 if times else 0.0
        velocity_1h[i] = sum(1 for s in times if t - s <= 3600)
        velocity_24h[i] = sum(1 for s in times if t - s <= 86400)

        if len(amts) >= 2:
            mu = np.mean(amts)
            sigma = np.std(amts) + 1e-6
            amt_zscore[i] = (a - mu) / sigma
        else:
            amt_zscore[i] = 0.0

        times.append(t)
        amts.append(a)
        card_times[c] = times
        card_amts[c] = amts

    df["velocity_1h"] = velocity_1h
    df["velocity_24h"] = velocity_24h
    df["amt_zscore"] = amt_zscore
    df["card_tenure_days"] = card_tenure
    return df


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ── Synthetic fallback ─────────────────────────────────────────────────────────

SYNTHETIC_VERBALIZABLE = {
    "amt":              "transaction amount in USD",
    "merchant":         "merchant name",
    "category":         "merchant category",
    "hour_of_day":      "hour of day (0–23)",
    "day_of_week":      "day of week (0=Mon)",
    "geo_distance_km":  "distance between cardholder and merchant (km)",
    "amt_zscore":       "transaction amount z-score vs card's historical mean/std",
    "velocity_1h":      "number of card transactions in the past 1 hour",
    "velocity_24h":     "number of card transactions in the past 24 hours",
    "card_tenure_days": "days since first observed transaction for this card",
}

CATEGORIES = ["grocery_pos", "gas_transport", "misc_net", "shopping_net",
               "shopping_pos", "food_dining", "travel", "entertainment", "health_fitness"]


def _generate_synthetic(n: int, seed: int) -> pd.DataFrame:
    """Faker + numpy synthetic generator with Sparkov-style schema."""
    from faker import Faker
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    n_cards = max(100, n // 50)
    card_ids = [fake.credit_card_number() for _ in range(n_cards)]

    # Per-card profile
    card_lat = rng.uniform(25, 48, n_cards)
    card_lon = rng.uniform(-122, -70, n_cards)
    card_mean_amt = rng.lognormal(3.5, 0.8, n_cards)  # ~$33 median
    card_std_amt = card_mean_amt * rng.uniform(0.3, 1.2, n_cards)

    # Sample transactions
    card_idx = rng.integers(0, n_cards, n)
    base_time = pd.Timestamp("2020-01-01")
    offsets = np.sort(rng.integers(0, 365 * 86400, n))
    timestamps = [base_time + pd.Timedelta(seconds=int(s)) for s in offsets]

    merchants = [fake.company()[:40] for _ in range(200)]
    merchant_lat = rng.uniform(25, 48, 200)
    merchant_lon = rng.uniform(-122, -70, 200)
    merch_idx = rng.integers(0, 200, n)

    amt_raw = np.abs(
        rng.normal(card_mean_amt[card_idx], card_std_amt[card_idx])
    ).clip(0.5, 10_000)

    # Inject fraud: card-testing bursts + geo anomalies
    is_fraud = np.zeros(n, dtype=np.int8)
    # Card-testing: many small transactions in short windows on 2% of cards
    fraud_cards = rng.choice(n_cards, size=max(1, n_cards // 50), replace=False)
    for fc in fraud_cards:
        mask = card_idx == fc
        idxs = np.where(mask)[0]
        if len(idxs) < 5:
            continue
        burst_start = rng.integers(0, max(1, len(idxs) - 10))
        burst = idxs[burst_start: burst_start + rng.integers(5, 15)]
        is_fraud[burst] = 1
        amt_raw[burst] = rng.uniform(0.5, 5.0, len(burst))  # tiny amounts

    # Geo anomaly: merchant far from cardholder
    geo_dist = _haversine(
        card_lat[card_idx], card_lon[card_idx],
        merchant_lat[merch_idx], merchant_lon[merch_idx],
    )
    far_mask = geo_dist > 1500
    is_fraud[far_mask] = np.where(rng.random(far_mask.sum()) < 0.4, 1, is_fraud[far_mask])

    df = pd.DataFrame({
        "cc_num":       [card_ids[i] for i in card_idx],
        "amt":          amt_raw.astype(np.float32),
        "merchant":     [merchants[i] for i in merch_idx],
        "category":     [CATEGORIES[i] for i in rng.integers(0, len(CATEGORIES), n)],
        "timestamp":    timestamps,
        "geo_distance_km": geo_dist.astype(np.float32),
        "is_fraud":     is_fraud,
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["_ts_sec"] = df["timestamp"].astype(np.int64) // 10**9
    df = _add_sparkov_behavioral(df)
    df["dataset_source"] = "synthetic"
    keep = list(SYNTHETIC_VERBALIZABLE.keys()) + ["is_fraud", "timestamp", "cc_num", "dataset_source"]
    return df[[c for c in keep if c in df.columns]]


# ── Time-based split ──────────────────────────────────────────────────────────

def _time_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1) -> pd.DataFrame:
    """Assign train/val/test by timestamp quantiles (no leakage)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    splits = np.full(n, "train", dtype=object)
    splits[train_end:val_end] = "val"
    splits[val_end:] = "test"
    df["train_split"] = splits
    return df


# ── Report helpers ─────────────────────────────────────────────────────────────

def _print_verbalizability_table(verb_map: dict, source: str) -> None:
    print(f"\n{'─'*65}")
    print(f"  Feature verbalizability table  ({source})")
    print(f"{'─'*65}")
    print(f"  {'Feature':<28} How rendered in English")
    print(f"{'─'*65}")
    for feat, desc in verb_map.items():
        print(f"  {feat:<28} {desc}")
    print(f"{'─'*65}\n")


def _print_class_balance(df: pd.DataFrame) -> None:
    total = len(df)
    for split in ["train", "val", "test"]:
        sub = df[df["train_split"] == split]
        n_fraud = sub["is_fraud"].sum()
        print(f"  {split:5s}: {len(sub):>8,} rows  |  fraud {n_fraud:>6,} ({100*n_fraud/len(sub):.2f}%)")


def _save_phase1_report(df: pd.DataFrame, verb_map: dict, source: str) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    report = {
        "dataset_source": source,
        "total_rows": len(df),
        "features": list(verb_map.keys()),
        "verbalizability": verb_map,
        "class_balance": {
            split: {
                "n": int((df["train_split"] == split).sum()),
                "n_fraud": int(df.loc[df["train_split"] == split, "is_fraud"].sum()),
            }
            for split in ["train", "val", "test"]
        },
    }
    out = REPORTS_DIR / "phase1_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Phase 1 report written to {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description="Phase 1: download + prep fraud data")
    parser.add_argument(
        "--dataset",
        default=cfg.get("dataset", "ieee-fraud-detection"),
        choices=["ieee-fraud-detection", "kartik2112/fraud-detection",
                 "ealtman2019/credit-card-transactions", "synthetic"],
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Subsample to this many rows (useful for quick tests)")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "transactions.parquet")
    args = parser.parse_args()

    limit = args.limit or cfg.get("dataset_size")
    raw_dir = DATA_DIR / "raw" / args.dataset.replace("/", "_")

    print(f"\n=== Phase 1 — dataset: {args.dataset} ===\n")

    if args.dataset == "ieee-fraud-detection":
        if not (raw_dir / "train_transaction.csv").exists():
            _kaggle_competition_download("ieee-fraud-detection", raw_dir)
        df = _load_ieee(raw_dir, limit)
        verb_map = IEEE_VERBALIZABLE

    elif args.dataset == "kartik2112/fraud-detection":
        if not any(raw_dir.rglob("*.csv")):
            _kaggle_dataset_download("kartik2112/fraud-detection", raw_dir)
        df = _load_sparkov(raw_dir, limit)
        verb_map = SPARKOV_VERBALIZABLE

    elif args.dataset == "ealtman2019/credit-card-transactions":
        if not any(raw_dir.rglob("*.csv")):
            _kaggle_dataset_download("ealtman2019/credit-card-transactions", raw_dir)
        df = _load_sparkov(raw_dir, limit)
        verb_map = SPARKOV_VERBALIZABLE

    else:  # synthetic
        n = limit or 100_000
        print(f"Generating {n:,} synthetic transactions …")
        df = _generate_synthetic(n, seed)
        verb_map = SYNTHETIC_VERBALIZABLE

    # Time-based split (no leakage)
    df = _time_split(df)

    # Write parquet
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nWrote {len(df):,} rows → {args.output}")

    # Report
    _print_verbalizability_table(verb_map, args.dataset)
    print("Class balance by split:")
    _print_class_balance(df)
    _save_phase1_report(df, verb_map, args.dataset)
    print(f"\nFeatures in parquet: {[c for c in df.columns if c not in ('timestamp','train_split','dataset_source')]}\n")


if __name__ == "__main__":
    main()
