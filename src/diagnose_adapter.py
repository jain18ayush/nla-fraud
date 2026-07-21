"""Standalone, read-only diagnostic for the InjectionAdapter (src/nla_common.py).

Does NOT touch src/roundtrip_eval.py (owned by a concurrent agent) and does NOT
retrain or modify any checkpoint. Test 1 runs on CPU only. Test 2 optionally
touches the GPU just to hold the embedding matrix in memory for cosine compares
(cheap, no forward pass through the 7B model, no other GPU contention expected
since we only allocate a [vocab, 3584] tensor and immediately free it if needed).

Usage:
    .venv/bin/python src/diagnose_adapter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nla_common import InjectionAdapter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ADAPTER_PATH = REPO / "checkpoints" / "av_sft" / "adapter.pt"
HOLDOUT_PATH = REPO / "checkpoints" / "av_sft" / "holdout.parquet"
OUT_PATH = REPO / "reports" / "phase7_adapter_diagnostic.json"

N_SAMPLE = 500
SEED = 0

AV_MODEL_SNAPSHOT_GLOB = (
    "/workspace/.hf_home/hub/models--kitft--nla-qwen2.5-7b-L20-av/snapshots/*"
)


def effective_rank(X: np.ndarray) -> dict:
    """Singular value spectrum of [N, D] output matrix, mean-centered? -> NOT
    centered: we want the raw rank of the injected-vector set as it actually
    appears to the AV (each row IS the injected token, not a residual).
    """
    # Economy SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    total = var.sum()
    cum = np.cumsum(var) / total
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    # "effective rank" a la Roy & Vetterli: exp(entropy(normalized singular values))
    p = var / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    eff_rank_entropy = float(np.exp(entropy))
    return {
        "singular_values_top20": [float(x) for x in S[:20]],
        "singular_values_tail5": [float(x) for x in S[-5:]],
        "n_components_for_90pct_variance": n90,
        "n_components_for_99pct_variance": n99,
        "effective_rank_entropy_based": eff_rank_entropy,
        "matrix_shape": list(X.shape),
        "nominal_max_rank": int(min(X.shape)),
    }


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading adapter checkpoint (CPU)...", flush=True)
    ckpt = torch.load(ADAPTER_PATH, map_location="cpu")
    d_target = ckpt["d_target"]
    injection_scale = ckpt["injection_scale"]
    d_model = ckpt["state_dict"]["proj.weight"].shape[0]
    print(f"d_target={d_target} d_model={d_model} injection_scale={injection_scale}")

    adapter = InjectionAdapter(d_target=d_target, d_model=d_model, injection_scale=injection_scale)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()

    print("Loading holdout activations...", flush=True)
    import pyarrow.parquet as pq

    table = pq.read_table(HOLDOUT_PATH)
    n_total = table.num_rows
    n = min(N_SAMPLE, n_total)
    idx = rng.choice(n_total, size=n, replace=False)
    idx.sort()

    act_col = table.column("activation_vector").to_pylist()
    acts = np.array([act_col[i] for i in idx], dtype=np.float64)  # [n, 128]
    print(f"Sampled {n}/{n_total} holdout rows. Raw activation shape: {acts.shape}")

    results = {
        "meta": {
            "adapter_path": str(ADAPTER_PATH),
            "holdout_path": str(HOLDOUT_PATH),
            "n_holdout_total": n_total,
            "n_sampled": n,
            "seed": SEED,
            "d_target": d_target,
            "d_model": d_model,
            "injection_scale": injection_scale,
        }
    }

    # ---------------- Test 1: geometry preservation ----------------
    with torch.no_grad():
        v_in = torch.from_numpy(acts).float()
        v_out = adapter(v_in).numpy().astype(np.float64)  # [n, 3584], already norm==injection_scale

    print("Output norms (should all == injection_scale): "
          f"mean={np.linalg.norm(v_out, axis=1).mean():.4f} "
          f"std={np.linalg.norm(v_out, axis=1).std():.6f}")

    N = n
    in_n = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-12)
    out_n = v_out / (np.linalg.norm(v_out, axis=1, keepdims=True) + 1e-12)
    sim_in_full = in_n @ in_n.T
    sim_out_full = out_n @ out_n.T
    iu = np.triu_indices(N, k=1)
    in_vals = sim_in_full[iu]
    out_vals = sim_out_full[iu]

    def stat_block(vals):
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p5": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
            "std": float(np.std(vals)),
            "n_pairs": int(vals.shape[0]),
        }

    input_cosine_stats = stat_block(in_vals)
    output_cosine_stats = stat_block(out_vals)

    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(in_vals, out_vals)
    spearman_r, spearman_p = spearmanr(in_vals, out_vals)

    eff_rank = effective_rank(v_out)
    eff_rank_input = effective_rank(acts)

    results["test1_geometry"] = {
        "input_pairwise_cosine": input_cosine_stats,
        "output_pairwise_cosine": output_cosine_stats,
        "geometry_preservation_correlation": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "note": "correlation between input-pair cosine and output-pair cosine "
                    "over the same i<j pairs; high+positive => geometry preserved "
                    "(possibly rescaled), low/near-zero => geometry scrambled, "
                    "not just shrunk.",
        },
        "output_effective_rank": eff_rank,
        "input_effective_rank_reference": eff_rank_input,
        "output_norm_check": {
            "mean": float(np.linalg.norm(v_out, axis=1).mean()),
            "std": float(np.linalg.norm(v_out, axis=1).std()),
            "expected": injection_scale,
        },
    }

    print(json.dumps(results["test1_geometry"]["input_pairwise_cosine"], indent=2))
    print(json.dumps(results["test1_geometry"]["output_pairwise_cosine"], indent=2))
    print("Pearson r:", pearson_r, "Spearman r:", spearman_r)

    # ---------------- Test 2: adapter output vs AV embedding space ----------------
    try:
        import glob

        snaps = glob.glob(AV_MODEL_SNAPSHOT_GLOB)
        if not snaps:
            raise FileNotFoundError("AV model snapshot not found under HF_HOME")
        snap = snaps[0]
        index_path = Path(snap) / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        embed_key = "model.embed_tokens.weight"
        shard_file = weight_map[embed_key]
        shard_path = Path(snap) / shard_file

        from safetensors import safe_open

        print(f"Lazy-loading {embed_key} from {shard_path} (CPU, partial safetensors read)...")
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            embed = f.get_tensor(embed_key).float()  # [vocab, 3584]

        vocab_size, d_model_embed = embed.shape
        print(f"Embedding matrix shape: {embed.shape}")

        embed_norms = embed.norm(dim=-1)
        real_token_norm_stats = {
            "mean": float(embed_norms.mean()),
            "median": float(embed_norms.median()),
            "p5": float(embed_norms.quantile(0.05)),
            "p95": float(embed_norms.quantile(0.95)),
            "std": float(embed_norms.std()),
        }

        # (a) random token embeddings: sample same n as adapter outputs
        rand_idx = rng.choice(vocab_size, size=n, replace=False)
        rand_embed = embed[rand_idx].numpy().astype(np.float64)
        rand_embed_n = rand_embed / (np.linalg.norm(rand_embed, axis=1, keepdims=True) + 1e-12)

        v_out_t = torch.from_numpy(v_out).float()
        out_n_t = v_out_t / (v_out_t.norm(dim=-1, keepdim=True) + 1e-12)

        # out_n_t: [n,3584], rand_embed_n: [n,3584] -> pairwise all-to-all n x n
        cos_to_random_full = out_n_t.numpy() @ rand_embed_n.T  # [n, n]
        cos_to_random_mean = float(cos_to_random_full.mean())
        cos_to_random_stats = {
            "mean": cos_to_random_mean,
            "median": float(np.median(cos_to_random_full)),
            "p5": float(np.percentile(cos_to_random_full, 5)),
            "p95": float(np.percentile(cos_to_random_full, 95)),
        }

        # (b) embedding matrix mean direction
        embed_mean_dir = embed.mean(dim=0)
        embed_mean_dir_n = embed_mean_dir / (embed_mean_dir.norm() + 1e-12)
        cos_to_mean_dir = (out_n_t @ embed_mean_dir_n).numpy()
        cos_to_mean_dir_stats = {
            "mean": float(cos_to_mean_dir.mean()),
            "median": float(np.median(cos_to_mean_dir)),
            "p5": float(np.percentile(cos_to_mean_dir, 5)),
            "p95": float(np.percentile(cos_to_mean_dir, 95)),
        }

        results["test2_embedding_space"] = {
            "av_model_snapshot": str(snap),
            "vocab_size": int(vocab_size),
            "embedding_d_model": int(d_model_embed),
            "real_token_l2_norm_stats": real_token_norm_stats,
            "injected_vector_l2_norm": injection_scale,
            "norm_ratio_injected_over_mean_real": float(injection_scale / real_token_norm_stats["mean"]),
            "cosine_adapter_output_vs_random_token_embeddings": cos_to_random_stats,
            "cosine_adapter_output_vs_embedding_mean_direction": cos_to_mean_dir_stats,
        }

        print(json.dumps(results["test2_embedding_space"], indent=2))

        del embed
    except Exception as e:
        results["test2_embedding_space"] = {"skipped": True, "reason": repr(e)}
        print(f"Test 2 skipped/failed: {e!r}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
