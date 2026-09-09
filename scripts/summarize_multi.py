"""Multi-model summary: aggregate stage2/3 outputs of bert-base / electra-small / distilbert."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs/multi_model"
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "bert-base-uncased": dict(s2="outputs/stage2", s3="outputs/stage3",
                              sub="output.dense", exc=("attention",)),
    "electra-small":      dict(s2="outputs/stage2_electra", s3="outputs/stage3_electra",
                               sub="output.dense", exc=("attention",)),
    "distilbert-base":    dict(s2="outputs/stage2_distilbert", s3="outputs/stage3_distilbert",
                               sub="ffn.lin2", exc=("attention",)),
}
GROUPS = ["baseline", "A", "B", "C"]

summary = {}
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, (mname, paths) in zip(axes, MODELS.items()):
    d = json.load(open(f"{paths['s2']}/spectra_summary.json"))["matrices"]
    sel = {k: v for k, v in d.items() if paths["sub"] in k and not any(x in k for x in paths["exc"])}
    R = json.load(open(f"{paths['s3']}/results.json"))
    summary[mname] = dict(
        spectra=dict(top10=float(np.mean([v["energy_top10"] for v in sel.values()])),
                     mid80=float(np.mean([v["energy_mid80"] for v in sel.values()])),
                     bot10=float(np.mean([v["energy_bot10"] for v in sel.values()]))),
        dev={g: R[g]["best"]["dev_acc"] for g in GROUPS},
        dev_loss_final={g: R[g]["final"]["dev_loss"] for g in GROUPS},
        invasionB=[float(x) for x in np.mean([np.array(inv["energy_by_decile_of_original_spectrum"])
                                              for inv in R["B"]["invasions"].values()], axis=0)],
    )
    colors = {"baseline": "#2c3e50", "A": "#c0392b", "B": "#27ae60", "C": "#8e44ad"}
    for g in GROUPS:
        h = R[g]["history"]
        ax.plot([x["epoch"] for x in h], [x["dev_acc"] for x in h], "o-", lw=1.2, ms=3,
                color=colors[g], label=g)
    ax.set_title(mname, fontsize=10)
    ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
axes[0].set_ylabel("RTE dev accuracy"); axes[0].legend(fontsize=8, ncol=2)
fig.suptitle("SVD ablation across models (10% RTE train, identical per-model hyper-params)", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/dev_curves.png", dpi=150)

# invasion decile comparison, group B vs baseline
fig, ax = plt.subplots(figsize=(8, 4))
for mname, c in zip(MODELS, ["#2c3e50", "#27ae60", "#e67e22"]):
    ax.plot(range(1, 11), summary[mname]["invasionB"], "o-", color=c, label=f"{mname} (group B)")
ax.plot(range(1, 11), np.ones(10) / 10, "k--", lw=0.8, label="uniform")
ax.set_xlabel("decile of original singular-value rank")
ax.set_ylabel("delta-energy fraction")
ax.set_title("Fine-tuning 'invasion dimensions', group B — all models spread to the tail")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/invasion_compare.png", dpi=150)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False)[:600])
print("saved:", OUT)
