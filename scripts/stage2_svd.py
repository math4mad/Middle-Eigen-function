"""Stage 2: load pretrained model, SVD-decompose weight matrices, plot spectra.

Hypothesis under test (AGETNTS.md): a model's generalization ability lives in the
*middle* of the singular-value spectrum ("loose spectrum"), not in the top few
singular values ("rigid skeleton").

This script:
  1. loads bert-base-uncased,
  2. computes the singular-value spectrum of every 2-D linear weight matrix
     (with focus on mlp.c_proj),
  3. reports spectral concentration: energy captured by top-10% / middle-80% /
     bottom-10% of singular values,
  4. saves plots + a JSON summary for the report.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel

MODEL_NAME = os.environ.get("MODEL_PATH", "models/AI-ModelScope/bert-base-uncased")
if not os.path.isdir(MODEL_NAME):
    MODEL_NAME = "AI-ModelScope/bert-base-uncased"  # fall back to ModelScope hub id
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage2")
# which matrices to profile: substring match minus excluded substrings (comma-separated)
AB_SUBSTR = os.environ.get("ABLATE_SUBSTR", "output.dense")
AB_EXCLUDE = tuple(x for x in os.environ.get("ABLATE_EXCLUDE", "attention,embeddings").split(",") if x)


def spectra_of(model):
    """Return {param_name: singular_values} for all 2-D weight matrices."""
    out = {}
    for name, p in model.named_parameters():
        w = p.detach().float().cpu().numpy()
        if w.ndim != 2:
            continue
        # skip embeddings & classifier/head: they are not "param matrices" here
        if "embeddings" in name:
            continue
        out[name] = np.linalg.svd(w, compute_uv=False)
    return out


def energy_band(s, frac_top=0.10, frac_bottom=0.10):
    """Energy (squared singular values) fractions in top / middle / bottom bands."""
    e = s ** 2
    total = e.sum()
    n = len(s)
    k_top = max(1, int(np.ceil(n * frac_top)))
    k_bot = max(1, int(np.ceil(n * frac_bottom)))
    # s is sorted descending by np.linalg.svd
    top = e[:k_top].sum() / total
    bot = e[-k_bot:].sum() / total
    mid = 1.0 - top - bot
    return float(top), float(mid), float(bot)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading {MODEL_NAME} ...", flush=True)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    spec = spectra_of(model)
    cproj = {k: v for k, v in spec.items()
             if AB_SUBSTR in k and not any(x in k for x in AB_EXCLUDE)}
    print(f"decomposed {len(spec)} matrices, {len(cproj)} target matrices (substr={AB_SUBSTR!r})")

    summary = {"model": MODEL_NAME, "n_matrices": len(spec), "matrices": {}}
    for name, s in spec.items():
        t, m, b = energy_band(s)
        summary["matrices"][name] = {
            "shape": list(s.shape),
            "s_max": float(s[0]),
            "s_med": float(np.median(s)),
            "s_min": float(s[-1]),
            "cond": float(s[0] / max(s[-1], 1e-12)),
            "energy_top10": t,
            "energy_mid80": m,
            "energy_bot10": b,
        }
    with open(f"{OUT_DIR}/spectra_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plot 1: spectra of the 12 mlp.c_proj matrices ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, s in cproj.items():
        ax.plot(np.arange(len(s)) / len(s), s / s[0], lw=1, alpha=0.7,
                label=f"L{int(name.split('.')[-4])}.target")
    ax.set_xlabel("normalized singular-value index")
    ax.set_ylabel("sigma / sigma_max")
    ax.set_yscale("log")
    ax.set_title("mlp.c_proj spectra (bert-base-uncased)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cproj_spectra.png", dpi=150)

    # ---- plot 2: energy bands across all matrices ----
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(summary["matrices"])
    top = [summary["matrices"][k]["energy_top10"] for k in keys]
    mid = [summary["matrices"][k]["energy_mid80"] for k in keys]
    bot = [summary["matrices"][k]["energy_bot10"] for k in keys]
    x = np.arange(len(keys))
    ax.stackplot(x, np.array(top) * 100, np.array(mid) * 100, np.array(bot) * 100,
                 labels=["top 10%", "middle 80%", "bottom 10%"], colors=["#c0392b", "#f1c40f", "#2980b9"])
    ax.set_xticks(x[::6], [keys[i] for i in x[::6]], rotation=90, fontsize=6)
    ax.set_ylabel("energy fraction (%)")
    ax.set_xlabel("weight matrix")
    ax.set_title("Where the spectral energy sits (Frobenius^2)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, len(keys))
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/energy_bands.png", dpi=150)

    # console report
    print(f"{'matrix':60s} {'top10E%':>8s} {'mid80E%':>8s} {'bot10E%':>8s}")
    for k in keys:
        d = summary["matrices"][k]
        print(f"{k:60s} {d['energy_top10']*100:8.2f} {d['energy_mid80']*100:8.2f} {d['energy_bot10']*100:8.2f}")
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
