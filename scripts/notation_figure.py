"""One schematic figure for the glossary page: what the spectrum, the bands, and the
top_E / mid_E / tail_E / TGT / RAND / SPEC notation actually point at.

Three panels, all drawn from REAL data in outputs/ (no synthetic curves):
  A  two real spectra (flat MLP vs steep attention) with decile bands + energy shares
  B  where the fine-tuning update ΔW writes its energy (clean run vs 50% junk run)
  C  which atoms the three deletion arms remove (positions from outputs/stage12_pure)

Writes outputs/glossary/schematic.png  (English-only labels: CJK glyphs are missing from
the bundled matplotlib fonts).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file

OUT = os.environ.get("OUT_DIR", "outputs/glossary")
QW = "models/models/Qwen--Qwen2.5-0.5B/snapshots/master/model.safetensors"
os.makedirs(OUT, exist_ok=True)

sd = load_file(QW)
spec = {}
for tag, key in [("MLP mlp.down_proj (L20)", "model.layers.20.mlp.down_proj.weight"),
                 ("attn q_proj (L20)", "model.layers.20.self_attn.q_proj.weight")]:
    s = np.linalg.svd(sd[key].float().numpy(), compute_uv=False)
    spec[tag] = s

SP = json.load(open("outputs/stage10/summary.json"))
runs = {r["f"]: r for r in SP["runs"] if r["cut"] == "none" and r["seed"] == 1}  # noqa
S12 = json.load(open("outputs/stage12_pure/results_partial.json"))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))

# ---------------- A: spectra + bands ----------------
colors = ["#b2182b", "#ef8a62", "#f4cbb0", "#d8daeb", "#b9c0d9", "#9ecae1",
          "#6baed6", "#4292c6", "#2171b5", "#084594"]
for j, (tag, s) in enumerate(spec.items()):
    a = ax[0] if j == 0 else ax[1]
    n = len(s)
    edges = np.linspace(0, n, 11).astype(int)
    a.plot(s, lw=1.6, color="k")
    for b in range(10):
        e = np.zeros(n)
        e[edges[b]:edges[b + 1]] = s.max() * 1.05
        a.fill_between(range(edges[b], min(edges[b + 1], n)), 0, e[edges[b]:edges[b + 1]] if b else e[edges[b]:edges[b+1]],
                       color=colors[b], alpha=0.18)
    tot = (s ** 2).sum()
    top = (s[:n // 10] ** 2).sum() / tot
    mid = (s[n // 10:n // 2] ** 2).sum() / tot
    tail = (s[n // 2:] ** 2).sum() / tot
    a.axvline(n // 10, color="#b2182b", ls="--", lw=1)
    a.axvline(n // 2, color="#2171b5", ls="--", lw=1)
    a.set_title(f"{tag}\nenergy: top 10% = {top*100:.0f}%  |  bands 2-5 = "
                f"{mid*100:.0f}%  |  bands 6-10 = {tail*100:.0f}%", fontsize=8.5)
    a.set_xlabel("atom index i (sigma sorted descending)")
    a.set_ylabel("sigma_i")
    a.grid(alpha=0.25)
    a.text(0.02 * n, s.max() * 0.75, "top_E\n(rigid skeleton)", fontsize=7, color="#b2182b")
    a.text(0.28 * n, s.max() * 0.55, "mid_E", fontsize=8, color="#6a51a3")
    a.text(0.72 * n, s.max() * 0.25, "tail_E\n(loose spectrum)", fontsize=7, color="#2171b5")

# ---------------- B: where ΔW writes its energy ----------------
axB = ax[2]
sens = {d["metric"]: d["by_f"] for d in SP["q2_sensitivity"]}
fs = sorted(float(k) for k in sens["top_E"])
x = np.arange(3)
cols = ["#bdbdbd", "#f4a261", "#e76f51", "#b2182b"]
for i, f in enumerate(fs):
    vals = [sens[k][str(f)] for k in ("top_E", "mid_E", "tail_E")]
    axB.bar(x + (i - 1.5) * 0.2, vals, width=0.2, color=cols[i], label=f"f = {f:g}")
axB.set_xticks(x)
axB.set_xticklabels(["top_E\n(band 1)", "mid_E\n(bands 2-5)", "tail_E\n(bands 6-10)"], fontsize=9)
axB.set_xlabel("share of dW energy per spectral region")
axB.set_ylabel("share of dW energy")
axB.set_title("Where the update writes itself\n(online monitor, mean of last 25% of steps)", fontsize=8.5)
axB.set_ylim(0, 0.68)
axB.annotate("mid_E rises monotonically with junk\n(rho=+0.83, d=+8.6); tail_E is NOT monotone\n"
             "=> junk prefers the MIDDLE bands", xy=(1.18, 0.47), xytext=(1.35, 0.60),
             fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=1))
axB.legend(fontsize=7, loc="upper left")
axB.grid(alpha=0.25, axis="y")
fig.subplots_adjust(left=0.05, right=0.98, wspace=0.32)

fig.suptitle("Notation made visible: spectra, bands, and where a fine-tuning update lands", y=1.04, fontsize=11)
fig.savefig(f"{OUT}/schematic.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT}/schematic.png")

# ---------------- C: deletion arms as a separate small figure ----------------
fig2, ax2 = plt.subplots(figsize=(6.4, 2.6))
s = spec["MLP mlp.down_proj (L20)"]
n = len(s)
ax2.plot(s, lw=1.2, color="k")
ax2.axvline(n // 2, color="#2171b5", ls="--", lw=1)
arm_pos = {}
for k, v in S12.items():
    if "removed_rank_frac" in v:
        arm_pos.setdefault(v["arm"], []).extend(v["removed_rank_frac"])
sty = {"tgt": ("o", "crimson", "TGT  (junk-fingerprint atoms)"),
       "rand": ("s", "seagreen", "RAND (band-matched random atoms)"),
       "spec": ("^", "steelblue", "SPEC  (largest sigma in bands 6-10)")}
for arm, (mk, col, lab) in sty.items():
    pos = np.array(arm_pos.get(arm, []))
    if len(pos):
        idx = (pos * n).astype(int)
        ax2.scatter(idx, s[idx], marker=mk, color=col, s=34,
                    label=f"{lab}  mean pos={pos.mean():.2f}")
ax2.set_xlabel("atom index i (0 = largest sigma)")
ax2.set_ylabel("sigma_i")
ax2.set_title("Stage 12 deletion arms: which atoms each coordinate removes (K=2 x 24 layers)")
ax2.legend(fontsize=7, loc="upper right")
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(f"{OUT}/deletion_arms.png", dpi=150)
print(f"wrote {OUT}/deletion_arms.png")
