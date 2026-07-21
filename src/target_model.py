"""
Phase 2a: FraudMLP — train a shallow MLP on the transactions parquet.

Architecture:
  embeddings(cats) ++ numerics
    -> Linear(d_in, 256) GELU Dropout   # l1
    -> Linear(256, 128)  GELU Dropout   # l2  <-- default hook target
    -> Linear(128, 128)  GELU           # l3
    -> Linear(128, 1)                   # fraud logit

Outputs:
  data/fraud_mlp.pt          — model weights
  data/mlp_artifacts.pkl     — preprocessing objects + feature schema
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def load_config() -> dict:
    with open(ROOT / "configs" / "experiment.yaml") as f:
        return yaml.safe_load(f)


# ── Feature schemas per dataset source ───────────────────────────────────────
# Anything not in CATEGORICALS is treated as numeric (filled with median, scaled).

_ANON_DOMAINS = {
    "protonmail.com", "guerrillamail.com", "yopmail.com", "dispostable.com",
    "mailnull.com", "sharklasers.com", "trashmail.com", "tempr.email", "cuvox.de",
    "anonaddy.com", "tutanota.com", "cock.li",
}
_FREE_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "live.com", "icloud.com", "me.com", "msn.com", "comcast.net",
    "yahoo.fr", "yahoo.co.uk", "yahoo.de", "yahoo.es", "yahoo.co.jp",
    "hotmail.co.uk", "hotmail.fr", "hotmail.de", "att.net", "verizon.net",
    "sbcglobal.net", "frontier.com",
}


def _email_group(domain) -> str:
    if pd.isna(domain):
        return "missing"
    d = str(domain).lower()
    if d in _ANON_DOMAINS:
        return "anonymous"
    if d in _FREE_DOMAINS:
        return "free"
    return "corporate"


def _add_ieee_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer additional verbalizable features for IEEE-CIS."""
    df = df.copy()
    df["p_email_group"] = df["P_emaildomain"].apply(_email_group)
    df["r_email_group"] = df["R_emaildomain"].apply(_email_group)
    # 1 if same domain, 0 if different, -1 if either missing
    p_null = df["P_emaildomain"].isna()
    r_null = df["R_emaildomain"].isna()
    df["email_match"] = np.where(
        p_null | r_null, -1.0,
        (df["P_emaildomain"] == df["R_emaildomain"]).astype(float)
    )
    return df


FEATURE_SCHEMAS = {
    "ieee-fraud-detection": {
        "categoricals": ["ProductCD", "card4", "card6", "P_emaildomain",
                         "R_emaildomain", "p_email_group", "r_email_group",
                         "DeviceType", "DeviceInfo"],
        "numerics": ["TransactionAmt", "addr1", "addr2", "dist1",
                     "hour_of_day", "day_of_week", "email_match",
                     "amt_zscore", "velocity_1h", "velocity_24h", "card_tenure_days"],
        "label": "is_fraud",
    },
    "sparkov": {
        "categoricals": ["merchant", "category"],
        "numerics": ["amt", "hour_of_day", "day_of_week", "geo_distance_km",
                     "amt_zscore", "velocity_1h", "velocity_24h", "card_tenure_days",
                     "lat", "long", "city_pop", "merch_lat", "merch_long", "age_at_txn"],
        "label": "is_fraud",
    },
    "synthetic": {
        "categoricals": ["merchant", "category"],
        "numerics": ["amt", "hour_of_day", "day_of_week", "geo_distance_km",
                     "amt_zscore", "velocity_1h", "velocity_24h", "card_tenure_days"],
        "label": "is_fraud",
    },
}
# IBM TabFormer reuses sparkov schema (same column names after _load_sparkov)
FEATURE_SCHEMAS["ealtman2019/credit-card-transactions"] = FEATURE_SCHEMAS["sparkov"]


# ── Preprocessing ─────────────────────────────────────────────────────────────

# High-cardinality categoricals get their vocabulary capped to the top-N values.
# Everything outside the top-N is remapped to "__rare__" before encoding.
CAT_VOCAB_CAP = 200

# Numeric columns with known outlier issues get hard-clipped before scaling.
NUMERIC_CLIP = {
    "amt_zscore": (-10.0, 10.0),
    "velocity_1h": (0.0, 200.0),
    "velocity_24h": (0.0, 1000.0),
}


class Preprocessor:
    """Fit on train, transform all splits. Persisted alongside the model."""

    def __init__(self, cat_cols: list[str], num_cols: list[str]):
        self.cat_cols = [c for c in cat_cols]
        self.num_cols = [c for c in num_cols]
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.top_cats: dict[str, set] = {}   # top-N values per high-cardinality col
        self.num_medians: dict[str, float] = {}
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        # Categoricals: cap high-cardinality, fill NA → "__missing__", LabelEncode
        for c in self.cat_cols:
            if c not in df.columns:
                continue
            vals = df[c].fillna("__missing__").astype(str)
            n_unique = vals.nunique()
            if n_unique > CAT_VOCAB_CAP:
                top = set(vals.value_counts().head(CAT_VOCAB_CAP).index)
                self.top_cats[c] = top
                vals = vals.apply(lambda v: v if v in top else "__rare__")
                print(f"  [{c}] capped {n_unique} → top-{CAT_VOCAB_CAP} + __rare__")
            le = LabelEncoder()
            le.fit(list(vals.unique()) + ["__missing__", "__unknown__", "__rare__"])
            self.label_encoders[c] = le

        # Numerics: fill NA with median, clip outliers, then StandardScale
        for c in self.num_cols:
            if c not in df.columns:
                continue
            col = df[c].copy()
            if c in NUMERIC_CLIP:
                lo, hi = NUMERIC_CLIP[c]
                col = col.clip(lo, hi)
            self.num_medians[c] = float(col.median())
        num_df = self._fill_and_clip(df)
        cols_present = [c for c in self.num_cols if c in df.columns]
        if cols_present:
            self.scaler.fit(num_df[cols_present].values)
        self._cols_present = cols_present
        return self

    def _fill_and_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c, med in self.num_medians.items():
            if c not in df.columns:
                continue
            col = df[c].copy()
            if c in NUMERIC_CLIP:
                lo, hi = NUMERIC_CLIP[c]
                col = col.clip(lo, hi)
            df[c] = col.fillna(med)
        return df

    def _apply_cat_cap(self, c: str, vals: pd.Series) -> pd.Series:
        if c in self.top_cats:
            top = self.top_cats[c]
            return vals.apply(lambda v: v if v in top else "__rare__")
        return vals

    def transform_cats(self, df: pd.DataFrame) -> np.ndarray:
        """Returns int64 array (n, n_cat_present)."""
        arrs = []
        for c in self.cat_cols:
            if c not in df.columns or c not in self.label_encoders:
                continue
            le = self.label_encoders[c]
            vals = df[c].fillna("__missing__").astype(str)
            vals = self._apply_cat_cap(c, vals)
            known = set(le.classes_)
            vals = vals.apply(lambda v: v if v in known else "__unknown__")
            arrs.append(le.transform(vals))
        if not arrs:
            return np.zeros((len(df), 0), dtype=np.int64)
        return np.stack(arrs, axis=1).astype(np.int64)

    def transform_nums(self, df: pd.DataFrame) -> np.ndarray:
        df = self._fill_and_clip(df)
        cols = self._cols_present
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        arr = df[cols].values.astype(np.float32)
        return self.scaler.transform(arr).astype(np.float32)

    @property
    def cat_vocab_sizes(self) -> list[int]:
        return [len(le.classes_) for le in self.label_encoders.values()]

    @property
    def n_num_features(self) -> int:
        return len(self._cols_present)


# ── Model ─────────────────────────────────────────────────────────────────────

class FraudMLP(nn.Module):
    def __init__(self, cat_vocab_sizes: list[int], n_num: int,
                 embed_dim: int = 8, dropout: float = 0.3):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, min(embed_dim, max(2, vocab // 2)))
            for vocab in cat_vocab_sizes
        ])
        cat_dim = sum(e.embedding_dim for e in self.embeddings)
        d_in = cat_dim + n_num

        self.l1 = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(dropout))
        self.l2 = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout))
        self.l3 = nn.Sequential(nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(128, 1)

    def forward(self, cats: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        parts = [e(cats[:, i]) for i, e in enumerate(self.embeddings)]
        if parts:
            x = torch.cat(parts + [nums], dim=1)
        else:
            x = nums
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.head(x).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_tensors(df: pd.DataFrame, prep: Preprocessor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cats = torch.tensor(prep.transform_cats(df), dtype=torch.long)
    nums = torch.tensor(prep.transform_nums(df), dtype=torch.float32)
    labels = torch.tensor(df["is_fraud"].values, dtype=torch.float32)
    return cats, nums, labels


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    mlp_cfg = cfg["mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    parquet = DATA_DIR / "transactions.parquet"
    print(f"Loading {parquet} …")
    df = pd.read_parquet(parquet)
    source = df["dataset_source"].iloc[0]
    schema = FEATURE_SCHEMAS.get(source, FEATURE_SCHEMAS["synthetic"])
    print(f"Dataset source: {source}  |  rows: {len(df):,}")

    # Filter to columns that exist (some optional cols may be absent)
    cat_cols = [c for c in schema["categoricals"] if c in df.columns]
    num_cols = [c for c in schema["numerics"] if c in df.columns]
    print(f"Cat features ({len(cat_cols)}): {cat_cols}")
    print(f"Num features ({len(num_cols)}): {num_cols}")

    if source == "ieee-fraud-detection":
        df = _add_ieee_features(df)
        # Refresh schema now that new columns exist
        cat_cols = [c for c in schema["categoricals"] if c in df.columns]
        num_cols = [c for c in schema["numerics"] if c in df.columns]
        print(f"After feature engineering:")
        print(f"  Cat features ({len(cat_cols)}): {cat_cols}")
        print(f"  Num features ({len(num_cols)}): {num_cols}")

    train_df = df[df["train_split"] == "train"].reset_index(drop=True)
    val_df   = df[df["train_split"] == "val"].reset_index(drop=True)
    test_df  = df[df["train_split"] == "test"].reset_index(drop=True)

    prep = Preprocessor(cat_cols, num_cols).fit(train_df)

    tr_cats, tr_nums, tr_labels = make_tensors(train_df, prep)
    va_cats, va_nums, va_labels = make_tensors(val_df, prep)
    te_cats, te_nums, te_labels = make_tensors(test_df, prep)

    bs = mlp_cfg["batch_size"]
    tr_loader = DataLoader(TensorDataset(tr_cats, tr_nums, tr_labels),
                           batch_size=bs, shuffle=True, num_workers=0)
    va_loader = DataLoader(TensorDataset(va_cats, va_nums, va_labels),
                           batch_size=bs * 4, shuffle=False, num_workers=0)
    te_loader = DataLoader(TensorDataset(te_cats, te_nums, te_labels),
                           batch_size=bs * 4, shuffle=False, num_workers=0)

    model = FraudMLP(
        cat_vocab_sizes=prep.cat_vocab_sizes,
        n_num=prep.n_num_features,
        dropout=mlp_cfg["dropout"],
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class-weighted BCE
    pos_weight = torch.tensor(
        [(train_df["is_fraud"] == 0).sum() / max(1, (train_df["is_fraud"] == 1).sum())],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=mlp_cfg["lr"],
        weight_decay=mlp_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, min_lr=1e-5,
    )

    best_val_auc = 0.0
    patience_count = 0
    best_state = None

    for epoch in range(1, mlp_cfg["epochs"] + 1):
        # Train
        model.train()
        total_loss = 0.0
        for cats_b, nums_b, labels_b in tr_loader:
            cats_b, nums_b, labels_b = cats_b.to(device), nums_b.to(device), labels_b.to(device)
            optimizer.zero_grad()
            logits = model(cats_b, nums_b)
            loss = criterion(logits, labels_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Val
        val_auc, val_pr = evaluate(model, va_loader, device)
        scheduler.step(1.0 - val_auc)
        print(f"Epoch {epoch:2d}  loss={total_loss/len(tr_loader):.4f}  "
              f"val_auc={val_auc:.4f}  val_pr={val_pr:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= mlp_cfg["early_stop_patience"]:
                print(f"Early stop at epoch {epoch}")
                break

    # Restore best
    model.load_state_dict(best_state)
    test_auc, test_pr = evaluate(model, te_loader, device)
    print(f"\n=== Test results ===")
    print(f"  AUC-ROC : {test_auc:.4f}")
    print(f"  AUC-PR  : {test_pr:.4f}")

    # Check bars
    bars = {"ieee-fraud-detection": 0.85, "sparkov": 0.90, "synthetic": 0.85}
    bar = bars.get(source, 0.85)
    if test_auc < bar:
        print(f"  [WARN] AUC-ROC {test_auc:.4f} < target {bar:.2f} — "
              f"consider revisiting feature engineering before Phase 3.")
    else:
        print(f"  [OK] AUC-ROC meets the >{bar:.2f} bar.")

    # Save — sklearn objects go to individual pickle files to avoid __main__
    # module-path issues; everything else is JSON.
    DATA_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), DATA_DIR / "fraud_mlp.pt")

    # Persist each sklearn LabelEncoder + scaler separately
    sklearn_dir = DATA_DIR / "mlp_sklearn"
    sklearn_dir.mkdir(exist_ok=True)
    for col, le in prep.label_encoders.items():
        with open(sklearn_dir / f"le_{col}.pkl", "wb") as f:
            pickle.dump(le, f)
    with open(sklearn_dir / "scaler.pkl", "wb") as f:
        pickle.dump(prep.scaler, f)

    meta = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cols_present_num": prep._cols_present,
        "cat_vocab_sizes": prep.cat_vocab_sizes,
        "n_num_features": prep.n_num_features,
        "num_medians": prep.num_medians,
        "top_cats": {k: list(v) for k, v in prep.top_cats.items()},
        "dataset_source": source,
        "model_config": {"dropout": mlp_cfg["dropout"]},
        "test_auc_roc": test_auc,
        "test_auc_pr": test_pr,
    }
    with open(DATA_DIR / "mlp_artifacts.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved: data/fraud_mlp.pt  data/mlp_artifacts.json  data/mlp_sklearn/")

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "phase2_model_report.json", "w") as f:
        json.dump({
            "dataset_source": source,
            "cat_features": cat_cols,
            "num_features": num_cols,
            "test_auc_roc": test_auc,
            "test_auc_pr": test_pr,
            "n_params": sum(p.numel() for p in model.parameters()),
        }, f, indent=2)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device
             ) -> tuple[float, float]:
    model.eval()
    all_logits, all_labels = [], []
    for cats_b, nums_b, labels_b in loader:
        logits = model(cats_b.to(device), nums_b.to(device)).cpu()
        all_logits.append(logits)
        all_labels.append(labels_b)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    auc = roc_auc_score(labels, probs)
    pr  = average_precision_score(labels, probs)
    return float(auc), float(pr)


def load_model_and_artifacts(device: torch.device | None = None
                              ) -> tuple[FraudMLP, "Preprocessor", dict]:
    """Utility used by collect_activations.py and later phases.

    Callers must apply source-specific feature engineering before calling
    make_tensors():
      if art['dataset_source'] == 'ieee-fraud-detection':
          df = _add_ieee_features(df)
    """
    with open(DATA_DIR / "mlp_artifacts.json") as f:
        art = json.load(f)

    # Reconstruct Preprocessor from saved components
    sklearn_dir = DATA_DIR / "mlp_sklearn"
    prep = Preprocessor(art["cat_cols"], art["num_cols"])
    for col in art["cat_cols"]:
        pkl = sklearn_dir / f"le_{col}.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                prep.label_encoders[col] = pickle.load(f)
    with open(sklearn_dir / "scaler.pkl", "rb") as f:
        prep.scaler = pickle.load(f)
    prep.num_medians = art["num_medians"]
    prep.top_cats = {k: set(v) for k, v in art["top_cats"].items()}
    prep._cols_present = art["cols_present_num"]

    model = FraudMLP(
        cat_vocab_sizes=art["cat_vocab_sizes"],
        n_num=art["n_num_features"],
        dropout=art["model_config"]["dropout"],
    )
    state = torch.load(DATA_DIR / "fraud_mlp.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    return model, prep, art


if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Override config path")
    args = parser.parse_args()
    train(cfg)
