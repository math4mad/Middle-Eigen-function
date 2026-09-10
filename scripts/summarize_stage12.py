"""Stage 12 summarizer — targeted (junk-fingerprint) vs band-matched random vs spectral-position deletion.

Reads outputs/stage12/results.json (from scripts/stage12_targeted_deletion.py) and answers:

  若 TGT − RAND > +0.03  ⇒ 污染可定位，但坐标是"污染方向"，不是"第几大 σ"
  若 TGT ≈ RAND ≈ SPEC   ⇒ 切割 = 能量/容量效应，"按大小切"可被"随便切"替代
  若三者都 ≤ 0 (vs noop) ⇒ 撤回删除式手术，只做在线监测

Writes outputs/stage12/{summary.json,arms.png} and prints the markdown table.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.environ.get("OUT_DIR", "outputs/stage12")
R = json.load(open(f"{OUT}/results.json" if os.path.exists(f"{OUT}/results.json")
                   else f"{OUT}/results_partial.json"))
runs = {k: v for k, v in R.items() if "best" in v and "arm" in v}
NOISE = 0.030

arms = sorted({v["arm"] for v in runs.values()})
by_frac = sorted({v["junk_frac"] for v in runs.values()})
rows, verdict = [], {}
for f in by_frac:
    sub = {v["arm"]: v for v in runs.values() if v["junk_frac"] == f}
    base = sub.get("noop")["best"]["dev_acc"] if "noop" in sub else np.nan
    for a in arms:
        if a not in sub:
            continue
        r = sub[a]
        fin = r.get("final") or r["history"][-1]
        rows.append({"f": f, "arm": a, "seed": r["seed"], "dev": r["best"]["dev_acc"],
                     "dev_last": fin["dev_acc"], "train": fin["train_acc"],
                     "dev_loss": fin["dev_loss"],
                     "delta_vs_noop": r["best"]["dev_acc"] - base,
                     "E_removed": r.get("energy_removed_mean", 0.0),
                     "mean_σrank_of_removed": float(np.mean(r.get("removed_rank_frac", [np.nan])))
                     if r.get("removed_rank_frac") else np.nan})
    if {"tgt", "rand", "noop"} <= set(sub):
        d = sub["tgt"]["best"]["dev_acc"] - sub["rand"]["best"]["dev_acc"]
        verdict[f"tgt_minus_rand_f{int(f*100)}"] = {
            "delta": float(d), "localizable": bool(d > NOISE),
            "dev_noop": float(sub["noop"]["best"]["dev_acc"]),
            "dev_tgt": float(sub["tgt"]["best"]["dev_acc"]),
            "dev_rand": float(sub["rand"]["best"]["dev_acc"])}
    if {"spec", "rand", "noop"} <= set(sub):
        verdict[f"spec_minus_rand_f{int(f*100)}"] = {
            "delta": float(sub["spec"]["best"]["dev_acc"] - sub["rand"]["best"]["dev_acc"])}
    if "noop" in sub:
        verdict[f"any_gain_f{int(f*100)}"] = {
            a: float(sub[a]["best"]["dev_acc"] - sub["noop"]["best"]["dev_acc"])
            for a in arms if a != "noop" and a in sub}

md = ["| f | arm | best dev | Δ vs noop | train | dev loss | 删掉的原子数×能量 | 原子平均谱位(0=最大σ) |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['f']:.2f} | {r['arm']} | {r['dev']:.3f} | {r['delta_vs_noop']:+.3f} | "
              f"{r['train']:.3f} | {r['dev_loss']:.2f} | K×{r['E_removed']*100:.2f}% | "
              f"{r['mean_σrank_of_removed']:.2f} |" if np.isfinite(r['mean_σrank_of_removed']) else
              f"| {r['f']:.2f} | {r['arm']} | {r['dev']:.3f} | {r['delta_vs_noop']:+.3f} | "
              f"{r['train']:.3f} | {r['dev_loss']:.2f} | — (不删) | — |")
text = "\n".join(md)
print(text)
print("\n=== verdict ===")
print(json.dumps(verdict, indent=2, ensure_ascii=False))
json.dump({"rows": rows, "verdict": verdict, "noise": NOISE},
          open(f"{OUT}/summary.json", "w"), indent=2, ensure_ascii=False)
open(f"{OUT}/stage12_table.md", "w").write(text + "\n")

if rows:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for a in arms:
        pts = [(r["f"], r["dev"]) for r in rows if r["arm"] == a]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=a)
    ax.axhspan(0.5 - NOISE, 0.5 + NOISE, color="grey", alpha=0.15, label="chance ± noise band")
    ax.set_xlabel("junk fraction f"); ax.set_ylabel("best dev acc")
    ax.set_title("Deletion arms: junk-fingerprint vs band-matched random vs spectral position")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/arms.png", dpi=150)
    print(f"wrote {OUT}/summary.json, stage12_table.md, arms.png")
