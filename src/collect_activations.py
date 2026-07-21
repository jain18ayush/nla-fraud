"""
Phase 2b: Collect activations from the trained FraudMLP.

For each of l2 and l3, forward-hooks capture the post-GELU activation vector.
Writes parquet files:
  data/activations_l2.parquet
  data/activations_l3.parquet

Each row: activation_vector (list[float]), activation_vector_normed,
           input features, fraud_score, label.

Also runs PCA + logistic probe and writes reports/activation_report.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def load_config() -> dict:
    with open(ROOT / "configs" / "experiment.yaml") as f:
        return yaml.safe_load(f)


# ── Activation collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_layer(
    model,
    loader: DataLoader,
    layer_name: str,
    device: torch.device,
    n_max: int,
) -> np.ndarray:
    """Forward-hook a named layer; return (n, d) float32 array."""
    activations = []

    def hook_fn(module, input, output):
        # output shape: (batch, dim) — GELU already applied inside the Sequential
        activations.append(output.detach().cpu().float().numpy())

    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(hook_fn)

    model.eval()
    collected = 0
    for cats_b, nums_b, _ in loader:
        model(cats_b.to(device), nums_b.to(device))
        collected += cats_b.size(0)
        if collected >= n_max:
            break

    handle.remove()
    arr = np.concatenate(activations, axis=0)
    return arr[:n_max]


# ── PCA report ────────────────────────────────────────────────────────────────

def pca_report(acts: np.ndarray, layer: str, n_components: int = 50) -> dict:
    n_comp = min(n_components, acts.shape[0] - 1, acts.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(acts)
    ev = np.cumsum(pca.explained_variance_ratio_)

    def components_for(threshold: float) -> int:
        idxs = np.where(ev >= threshold)[0]
        return int(idxs[0] + 1) if len(idxs) else n_comp

    c90 = components_for(0.90)
    c95 = components_for(0.95)
    c99 = components_for(0.99)
    print(f"\n  [{layer}] PCA explained variance:")
    print(f"    90% → {c90} components")
    print(f"    95% → {c95} components")
    print(f"    99% → {c99} components  (out of {acts.shape[1]} dims)")
    if c99 <= 5:
        print(f"    [NOTE] Only {c99} components for 99% var — activations are "
              f"nearly low-rank. Explanations will be limited in richness. "
              f"Consider using l2 over l3 if this is the l3 layer.")
    return {
        "layer": layer,
        "n_dims": acts.shape[1],
        "n_samples_pca": acts.shape[0],
        "components_90pct": c90,
        "components_95pct": c95,
        "components_99pct": c99,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def probe_report(acts: np.ndarray, labels: np.ndarray, layer: str) -> dict:
    # Subsample for speed
    n = min(50_000, len(acts))
    idx = np.random.choice(len(acts), n, replace=False)
    X, y = acts[idx], labels[idx]
    split = int(0.8 * n)
    lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    lr.fit(X[:split], y[:split])
    probs = lr.predict_proba(X[split:])[:, 1]
    auc = roc_auc_score(y[split:], probs)
    print(f"  [{layer}] Logistic probe AUC-ROC: {auc:.4f}")
    return {"layer": layer, "probe_auc_roc": auc, "n_probe_samples": n}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    act_cfg = cfg["activations"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=act_cfg["n_collect"],
                        help="Max rows to collect activations for")
    parser.add_argument("--layers", nargs="+", default=["l2", "l3"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model + data
    from target_model import load_model_and_artifacts, make_tensors, FEATURE_SCHEMAS, _add_ieee_features

    model, prep, art = load_model_and_artifacts(device)
    source = art["dataset_source"]

    parquet = DATA_DIR / "transactions.parquet"
    print(f"Loading {parquet} …")
    df = pd.read_parquet(parquet)
    # Apply same feature engineering used during training
    if source == "ieee-fraud-detection":
        df = _add_ieee_features(df)
    # Use all splits for the activation corpus (but record split so callers can filter)
    df = df.reset_index(drop=True)

    n_collect = min(args.n, len(df))
    df_sub = df.iloc[:n_collect].copy()
    print(f"Collecting activations for {n_collect:,} rows …")

    cats_t, nums_t, labels_t = make_tensors(df_sub, prep)
    loader = DataLoader(
        TensorDataset(cats_t, nums_t, labels_t),
        batch_size=1024, shuffle=False, num_workers=0,
    )

    # Get fraud scores in one pass before hooking layers
    scores = []
    model.eval()
    with torch.no_grad():
        for cats_b, nums_b in DataLoader(
            TensorDataset(cats_t, nums_t), batch_size=1024, shuffle=False
        ):
            logits = model(cats_b.to(device), nums_b.to(device)).cpu().numpy()
            scores.append(1 / (1 + np.exp(-logits)))
    fraud_scores = np.concatenate(scores)

    pca_reports = []
    probe_reports = []

    for layer_name in args.layers:
        print(f"\n=== Collecting {layer_name} activations ===")
        acts = collect_layer(model, loader, layer_name, device, n_collect)
        print(f"  Shape: {acts.shape}")

        # L2-normalise
        norms = np.linalg.norm(acts, axis=1, keepdims=True).clip(1e-8)
        acts_normed = acts / norms

        # PCA + probe
        pca_r = pca_report(acts_normed, layer_name, act_cfg["pca_components"])
        probe_r = probe_report(acts_normed, labels_t.numpy()[:n_collect], layer_name)
        pca_reports.append(pca_r)
        probe_reports.append(probe_r)

        # Build output dataframe
        feature_cols = prep.cat_cols + prep.num_cols
        feat_df = df_sub[[c for c in feature_cols if c in df_sub.columns]].copy()

        out_df = feat_df.copy()
        out_df["activation_vector"] = [row.tolist() for row in acts]
        out_df["activation_vector_normed"] = [row.tolist() for row in acts_normed]
        out_df["fraud_score"] = fraud_scores[:n_collect]
        out_df["label"] = labels_t.numpy()[:n_collect].astype(int)
        out_df["train_split"] = df_sub["train_split"].values

        out_path = DATA_DIR / f"activations_{layer_name}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"  Written → {out_path}  ({len(out_df):,} rows)")

    # Save report
    REPORTS_DIR.mkdir(exist_ok=True)
    report = {
        "dataset_source": source,
        "n_collected": n_collect,
        "hook_layer_default": cfg["activations"]["hook_layer"],
        "pca": pca_reports,
        "probe": probe_reports,
    }
    out = REPORTS_DIR / "activation_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nActivation report → {out}")

    # Plot if matplotlib available
    _try_plot_pca(pca_reports)


def _try_plot_pca(pca_reports: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    REPORTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(pca_reports), figsize=(6 * len(pca_reports), 4))
    if len(pca_reports) == 1:
        axes = [axes]
    for ax, r in zip(axes, pca_reports):
        ev = np.cumsum(r["explained_variance_ratio"])
        ax.plot(range(1, len(ev) + 1), ev, marker=".")
        for thresh, c in [(0.90, r["components_90pct"]),
                          (0.95, r["components_95pct"]),
                          (0.99, r["components_99pct"])]:
            ax.axhline(thresh, color="gray", linestyle="--", linewidth=0.8)
            ax.axvline(c, color="red", linestyle=":", linewidth=0.8,
                       label=f"{int(thresh*100)}%→{c}")
        ax.set_title(f"PCA cumulative variance — {r['layer']}")
        ax.set_xlabel("# components")
        ax.set_ylabel("cumulative explained variance")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = REPORTS_DIR / "activation_pca.png"
    plt.savefig(out, dpi=120)
    print(f"PCA plot → {out}")


if __name__ == "__main__":
    main()
