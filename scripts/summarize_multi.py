"""Multi-model summary: aggregate stage2/3 outputs of bert-base / electra-small / distilbert / roberta (+ energy-preserved variants)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs/multi_model"
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "bert-base-uncased": dict(s2="outputs/stage2", s3=["outputs/stage3", "outputs/stage3_energy"],
                              sub="output.dense", exc=("attention",)),
    "roberta-base":       dict(s2="outputs/stage2_roberta", s3=["outputs/stage3_roberta"],
                               sub="output.dense", exc=("attention",)),
    "electra-small":      dict(s2="outputs/stage2_electra", s3=["outputs/stage3_electra"],
                               sub="output.dense", exc=("attention",)),
    "distilbert-base":    dict(s2="outputs/stage2_distilbert", s3=["outputs/stage3_distilbert"],
                               sub="ffn.lin2", exc=("attention",)),
    "qwen2.5-0.5B+LoRA":  dict(s2="outputs/stage2_qwen", s3=["outputs/stage3_qwen"],
                               sub="mlp.down_proj", exc=("attention",)),
}
GROUPS = ["baseline", "A", "B", "C", "A_e", "B_e"]

summary = {}
fig, axes = plt.subplots(1, 5, figsize=(19, 4), sharey=True)
for ax, (mname, paths) in zip(axes, MODELS.items()):
    d = json.load(open(f"{paths['s2']}/spectra_summary.json"))["matrices"]
    sel = {k: v for k, v in d.items() if paths["sub"] in k and not any(x in k for x in paths["exc"])}
    R = {}
    for p in paths["s3"]:
        f = f"{p}/results.json"
        if os.path.exists(f):
            R.update(json.load(open(f)))
    summary[mname] = dict(
        spectra=dict(top10=float(np.mean([v["energy_top10"] for v in sel.values()])),
                     mid80=float(np.mean([v["energy_mid80"] for v in sel.values()])),
                     bot10=float(np.mean([v["energy_bot10"] for v in sel.values()]))),
        dev={g: R[g]["best"]["dev_acc"] for g in GROUPS if g in R},
        dev_loss_final={g: R[g]["final"]["dev_loss"] for g in GROUPS if g in R},
        train_acc_best={g: R[g]["best"]["train_acc"] for g in GROUPS if g in R},
        invasion={g: [float(x) for x in np.mean([np.array(i["energy_by_decile_of_original_spectrum"])
                                                 for i in R[g]["invasions"].values()], axis=0)]
                  for g in GROUPS if g in R},
    )
    colors = {"baseline": "#2c3e50", "A": "#c0392b", "B": "#27ae60", "C": "#8e44ad",
              "A_e": "#e74c3c", "B_e": "#2ecc71"}
    for g in GROUPS:
        if g not in R:
            continue
        h = R[g]["history"]
        ax.plot([x["epoch"] for x in h], [x["dev_acc"] for x in h], "o-", lw=1.2, ms=3,
                color=colors[g], ls="--" if g.endswith("_e") else "-", label=g)
    ax.set_title(mname, fontsize=10)
    ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
axes[0].set_ylabel("RTE dev accuracy"); axes[0].legend(fontsize=8, ncol=2)
fig.suptitle("SVD ablation across models (10% RTE train, identical per-model hyper-params; dashed = energy-preserved)", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/dev_curves.png", dpi=150)

# invasion decile comparison: baseline vs B vs B_e for bert & roberta
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, mname in zip(axes, ["bert-base-uncased", "roberta-base"]):
    s = summary[mname]["invasion"]
    for g, c in [("baseline", "#2c3e50"), ("B", "#27ae60"), ("B_e", "#2ecc71")]:
        if g in s:
            ax.plot(range(1, 11), s[g], "o-", color=c, label=g)
    ax.axhline(0.1, color="k", ls="--", lw=0.8)
    ax.set_title(mname); ax.set_xlabel("decile of original spectrum rank")
axes[0].set_ylabel("ΔW energy fraction"); axes[0].legend(fontsize=8)
fig.suptitle("Where fine-tuning invades (group B, plain vs energy-preserved); Qwen row uses LoRA ΔW",
             y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/invasion_compare.png", dpi=150)

# Qwen LoRA invasion (all groups) — position-neutral deltas
RQ = json.load(open("outputs/stage3_qwen/results.json"))
fig, ax = plt.subplots(figsize=(7, 3.5))
for g in ["baseline", "A", "B", "C"]:
    dec = np.mean([np.array(i["energy_by_decile_of_original_spectrum"]) for i in RQ[g]["invasions"].values()], axis=0)
    ax.plot(range(1, 11), dec / dec.sum(), "o-", ms=4, label=g)
ax.axhline(0.1, color="k", ls="--", lw=0.8)
ax.set_xlabel("decile of original down_proj spectrum rank")
ax.set_ylabel("LoRA ΔW energy fraction")
ax.set_title("Qwen2.5-0.5B: LoRA updates are spectrally position-neutral")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/invasion_qwen.png", dpi=150)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
for m in MODELS:
    print(m, json.dumps(summary[m]["dev"], ensure_ascii=False))
print("saved:", OUT)
