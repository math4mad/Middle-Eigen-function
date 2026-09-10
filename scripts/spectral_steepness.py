"""Spectral steepness diagnostic — how much does the σ *ordering* actually carry?

For a weight matrix W with singular values s_1 ≥ … ≥ s_n we report
  cond      s_1 / s_n                          (spectral dynamic range)
  cv        std(s) / mean(s)                   (dispersion of the spectrum)
  rel_perm  E‖U diag(π(s)) Vᵀ − W‖_F / ‖W‖_F   (how much the matrix changes if you keep the
                                               singular VALUES but randomly re-pair them with
                                               directions — averaged over 20 permutations)
  E_top10   fraction of spectral energy in the top 10% of σ

rel_perm is the operative number: rel_perm → 0 means the spectrum is so flat that "cutting by
magnitude" is indistinguishable from "cutting at random"; rel_perm → √2 means the ordering is a
strong, energy-bearing coordinate.  This is the quantitative version of the project's
"按大小切割是否欠妥" question (stage/stage11.md).
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file

MODELS = {
    "Qwen2.5-0.5B": ("models/models/Qwen--Qwen2.5-0.5B/snapshots/master/model.safetensors",
                     ["mlp.down_proj", "mlp.gate_proj", "q_proj", "o_proj"]),
    "Qwen2.5-1.5B": ("models/models/Qwen--Qwen2.5-1.5B/snapshots/master/model.safetensors",
                     ["mlp.down_proj", "mlp.gate_proj", "q_proj", "o_proj"]),
    "bert-base": ("models/models/AI-ModelScope--bert-base-uncased/snapshots/master/model.safetensors",
                  ["attention.output.dense", "intermediate.dense", "output.dense"]),
    "roberta-base": ("models/models/AI-ModelScope--roberta-base/snapshots/master/model.safetensors",
                     ["attention.output.dense", "intermediate.dense", "output.dense"]),
    "distilbert": ("models/models/AI-ModelScope--distilbert-base-uncased/snapshots/master/model.safetensors",
                   ["attention.output.dense", "intermediate.dense"]),
    "electra-small": ("models/models/google--electra-small-discriminator/snapshots/master/model.safetensors",
                      ["attention.output.dense", "intermediate.dense", "output.dense"]),
}
OUT = os.environ.get("OUT_DIR", "outputs/stage11")
N_PERM = 20
MAX_LAYERS = 12


def steepness(W, rng):
    s = np.linalg.svd(W, compute_uv=False).astype(np.float64)
    n = len(s)
    rel = []
    for _ in range(N_PERM):
        sp = rng.permutation(s)
        rel.append(np.linalg.norm(sp - s) / np.linalg.norm(s))
    e = s ** 2
    return {"n": int(n), "cond": float(s[0] / max(s[-1], 1e-12)),
            "cv": float(s.std() / s.mean()),
            "rel_perm": float(np.mean(rel)), "rel_perm_sd": float(np.std(rel)),
            "E_top10": float(e[:max(1, n // 10)].sum() / e.sum()),
            "r_eff": float(np.exp(-(e / e.sum() * np.log(e / e.sum() + 1e-20)).sum()))}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {}
    for model, (path, pats) in MODELS.items():
        if not os.path.exists(path):
            print(f"skip {model} (no {path})")
            continue
        sd = load_file(path)
        res[model] = {}
        for pat in pats:
            keys = [k for k in sd if pat in k and k.endswith(".weight")][:MAX_LAYERS]
            if not keys:
                continue
            rows = [steepness(sd[k].float().numpy(), rng) for k in keys]
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            agg["n_layers_sampled"] = len(rows)
            agg["shape"] = list(sd[keys[0]].shape)
            res[model][pat] = agg
            print(f"{model:14s} {pat:24s} shape={tuple(agg['shape'])} "
                  f"cond={agg['cond']:9.1f} cv={agg['cv']:.3f} "
                  f"rel_perm={agg['rel_perm']:.3f} E_top10={agg['E_top10']:.2f} "
                  f"r_eff={agg['r_eff']:.0f}")
    json.dump(res, open(f"{OUT}/steepness.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4))
    labels, vals, pats = [], [], []
    for m, d in res.items():
        for p, a in d.items():
            labels.append(f"{m}\n{p}")
            vals.append(a["rel_perm"])
            pats.append(a["cv"])
    order = np.argsort(vals)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(vals)))
    ax.barh([labels[i] for i in order], [vals[i] for i in order],
            color=[colors[i] for i in order])
    for i, idx in enumerate(order):
        ax.text(vals[idx] + 0.01, i, f"cv={pats[idx]:.2f}", va="center", fontsize=7)
    ax.axvline(np.sqrt(2), color="k", ls=":", lw=1)
    ax.text(np.sqrt(2) - 0.02, 0.5, "√2 = ordering fully load-bearing", fontsize=7, ha="right")
    ax.set_xlabel("rel_perm = E||U diag(pi(s)) V^T - W||_F / ||W||_F  (random re-pairing of sigma)")
    ax.set_title("Spectral steepness = how much the sigma ORDERING carries")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/steepness.png", dpi=150)
    print(f"\nwrote {OUT}/steepness.json, steepness.png")


if __name__ == "__main__":
    main()
