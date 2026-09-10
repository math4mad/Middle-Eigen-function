"""Stage 9 summarizer — the four robustness checks for the L6 (deep top-spectrum) finding.

Reads:
  outputs/stage9_seed{1,2,3}/results.json   ① multi-seed replication  (baseline vs TAIL6)
  outputs/stage9_boundary/results.json      ② depth boundary          (TAIL3 / TAIL6 / TAIL12)
  outputs/stage9_1p5b/results.json          ③ scale                   (1.5B: baseline/TAIL7/TAIL14)
  outputs/stage9_30pct/results.json         ④ data size 30%           (baseline vs TAIL6)
  outputs/stage3_qwen + stage5_qwen_deep    reference seed-42 run (baseline / L6 == TAIL6)

Writes outputs/stage9/{summary.json, stage9_table.md, deltas.png, curves.png}
and prints the markdown table to stdout.

判据（stage/stage9.md）:
  ① 3/4 个 seed 上 TAIL6−baseline 同号 且 均值 > +0.03  → 效应成立
  ② TAIL3 ≈ TAIL6 > TAIL12（倒 U）
  ③ 1.5B 上 TAIL7 复现同样增益
  ④ 30% 数据下优势缩小
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("OUT_ROOT", "outputs")
OUT = f"{ROOT}/stage9"
DEV_N = 277          # RTE dev size → 1 sample = 0.36%; binomial sd ≈ 3%
NOISE = np.sqrt(0.5 * 0.5 / DEV_N)   # ≈ 0.030


def load(path):
    p = f"{ROOT}/{path}/results.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def acc(res, g, key="best"):
    return res[g][key]["dev_acc"] if res and g in res else None


def gap(res, g):
    """generalization gap at final epoch: train_acc - dev_acc (and loss gap)."""
    if not res or g not in res:
        return None
    f = res[g]["final"]
    return {"train_acc": f["train_acc"], "dev_acc": f["dev_acc"],
            "acc_gap": f["train_acc"] - f["dev_acc"],
            "loss_gap": f["train_loss"] - f["dev_loss"]}


def ablated_layers(res, g):
    """how many down_proj matrices were actually cut, and mean energy kept on them."""
    ab = res[g]["ablation"]
    cut = [v for v in ab.values() if v.get("kept") != "intact"]
    if not cut:
        return "0/24", 1.0
    e = float(np.mean([v["energy_kept"] for v in cut]))
    return f"{len(cut)}/24", e


rows, checks = [], {}

# ---------- ① multi-seed ----------
seed_rows = []
# seed-42 reference for the 10% subset: baseline came from stage3_qwen, "L6"(=TAIL6) from stage5
_s3, _s5 = load("stage3_qwen"), load("stage5_qwen_deep")
ref = None
if _s3 and _s5 and "baseline" in _s3 and "L6" in _s5:
    ref = {"baseline": _s3["baseline"], "TAIL6": _s5["L6"]}
seeds = {}
for s, d in [(42, ref), (1, load("stage9_seed1")), (2, load("stage9_seed2")), (3, load("stage9_seed3"))]:
    if not d:
        continue
    b, t = acc(d, "baseline"), acc(d, "TAIL6")
    if b is None or t is None:
        continue
    seeds[s] = {"baseline": b, "TAIL6": t, "delta": t - b,
                "delta_loss": d["TAIL6"]["final"]["dev_loss"] - d["baseline"]["final"]["dev_loss"]}
if seeds:
    ds = [v["delta"] for v in seeds.values()]
    checks["1_multi_seed"] = {
        "per_seed": seeds, "deltas": ds, "mean_delta": float(np.mean(ds)),
        "n_positive": int(sum(d > 0 for d in ds)), "n": len(ds),
        "mean_delta_gt_0.03": bool(np.mean(ds) > 0.03),
        "sign_consistent_3of4": bool(sum(d > 0 for d in ds) >= 3 and len(ds) >= 4),
        "noise_sd": float(NOISE),
    }
    for s, v in seeds.items():
        rows.append(("①多seed", f"seed{s}", "baseline", v["baseline"], "", ""))
        rows.append(("①多seed", f"seed{s}", "TAIL6", v["TAIL6"], f"{v['delta']:+.3f}", ""))

# ---------- ② boundary ----------
bnd = load("stage9_boundary")
if bnd:
    base = acc(bnd, "baseline") or (acc(ref, "baseline") if ref else None)
    prof = {}
    for g in ("TAIL3", "TAIL6", "TAIL12"):
        if g in bnd:
            a = acc(bnd, g)
            prof[g] = {"dev_acc": a, "delta_vs_baseline": (a - base) if base else None,
                       "layers_cut": ablated_layers(bnd, g)[0], "energy_kept": ablated_layers(bnd, g)[1],
                       "gap": gap(bnd, g)}
    ordered = [prof[g]["dev_acc"] for g in ("TAIL3", "TAIL6", "TAIL12") if g in prof]
    checks["2_boundary"] = {
        "profile": prof, "baseline": base,
        "inverted_U": bool(len(ordered) == 3 and ordered[1] == max(ordered) and ordered[2] < ordered[1]),
    }
    for g, v in prof.items():
        rows.append(("②边界", "10%", g, v["dev_acc"],
                     f"{v['delta_vs_baseline']:+.3f}" if v["delta_vs_baseline"] is not None else "",
                     f"{v['layers_cut']} cut, E={v['energy_kept']:.2f}"))

# ---------- ③ scale (1.5B) ----------
big = load("stage9_1p5b")
if big and "baseline" in big:
    b = acc(big, "baseline")
    prof = {}
    for g in ("TAIL7", "TAIL14"):
        if g in big:
            n_layers = len(big[g]["ablation"])
            cut = [v for v in big[g]["ablation"].values() if v.get("kept") != "intact"]
            prof[g] = {"dev_acc": acc(big, g), "delta_vs_baseline": acc(big, g) - b,
                       "layers_cut": f"{len(cut)}/{n_layers}",
                       "energy_kept": float(np.mean([v["energy_kept"] for v in cut])) if cut else 1.0,
                       "gap": gap(big, g)}
    checks["3_scale"] = {"profile": prof, "baseline": b,
                         "TAIL7_gain": bool(prof.get("TAIL7", {}).get("delta_vs_baseline", -9) > 0.03)}
    for g, v in prof.items():
        rows.append(("③规模1.5B", "10%", g, v["dev_acc"], f"{v['delta_vs_baseline']:+.3f}",
                     f"{v['layers_cut']} cut, E={v['energy_kept']:.2f}"))
    rows.append(("③规模1.5B", "10%", "baseline", b, "", ""))

# ---------- ④ data size ----------
d30 = load("stage9_30pct")
if d30 and "baseline" in d30 and "TAIL6" in d30:
    delta30 = acc(d30, "TAIL6") - acc(d30, "baseline")
    delta10 = checks["1_multi_seed"]["per_seed"].get(42, {}).get("delta") if "1_multi_seed" in checks else None
    checks["4_data_size"] = {"delta_10pct": delta10, "delta_30pct": delta30,
                             "baseline_30": acc(d30, "baseline"), "TAIL6_30": acc(d30, "TAIL6"),
                             "advantage_shrinks": bool(delta10 is not None and delta30 < delta10)}
    for g in ("baseline", "TAIL6"):
        rows.append(("④数据30%", "30%", g, acc(d30, g),
                     f"{delta30:+.3f}" if g == "TAIL6" else "", ""))

# ---------- ⑤ pooled paired statistics (TAIL6/TAIL7 − baseline, same setting) ----------
def last3(res, g):
    h = res[g]["history"]
    return float(np.mean([e["dev_acc"] for e in h[-3:]]))

PAIRS = {}
if ref:
    PAIRS["0.5B 10% seed42"] = ref
for s in (1, 2, 3):
    r = load(f"stage9_seed{s}")
    if r:
        PAIRS[f"0.5B 10% seed{s}"] = r
if d30:
    PAIRS["0.5B 30% seed42"] = d30
if big and "baseline" in big and "TAIL7" in big:
    PAIRS["1.5B 10% seed42"] = big

stat_rows = []
for name, res in PAIRS.items():
    g = "TAIL7" if name.startswith("1.5B") else "TAIL6"
    if g not in res or "baseline" not in res:
        continue
    stat_rows.append({
        "setting": name, "group": g,
        "delta_best": res[g]["best"]["dev_acc"] - res["baseline"]["best"]["dev_acc"],
        "delta_last3": last3(res, g) - last3(res, "baseline"),
        "delta_devloss": res[g]["final"]["dev_loss"] - res["baseline"]["final"]["dev_loss"],
    })

def paired_stats(key, subset=None):
    from scipy import stats
    x = np.array([r[key] for r in stat_rows if subset is None or subset(r["setting"])])
    if len(x) < 2:
        return {"n": int(len(x))}
    t, p = stats.ttest_1samp(x, 0.0)
    ci = stats.bootstrap((x,), np.mean, paired=False, confidence_level=0.95,
                        random_state=0, method="percentile")
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "se": float(x.std(ddof=1) / np.sqrt(len(x))), "t": float(t), "p": float(p),
            "ci95": [float(ci.confidence_interval.low), float(ci.confidence_interval.high)],
            "positive": int((x > 0).sum())}

STRICT = lambda s: s.startswith("0.5B 10%")          # same data subset, only seed varies
checks["5_paired"] = {
    "per_pair": stat_rows,
    "strict_same_data_4seeds": {k: paired_stats(k, STRICT) for k in ("delta_best", "delta_last3", "delta_devloss")},
    "pooled_all_settings": {k: paired_stats(k) for k in ("delta_best", "delta_last3", "delta_devloss")},
}

n_strict = checks["5_paired"]["strict_same_data_4seeds"]["delta_best"]["n"]
mean_strict = checks["5_paired"]["strict_same_data_4seeds"]["delta_best"]["mean"]
p_pooled = checks["5_paired"]["pooled_all_settings"]["delta_best"]["p"]
checks["verdict"] = {
    "tier": ("all-four-pass" if all([checks.get('1_multi_seed', {}).get('sign_consistent_3of4'),
                                     checks.get('1_multi_seed', {}).get('mean_delta_gt_0.03'),
                                     checks.get('2_boundary', {}).get('inverted_U'),
                                     checks.get('3_scale', {}).get('TAIL7_gain')]) else
             "small-model-only" if checks.get("1_multi_seed", {}).get("sign_consistent_3of4") else
             "noise-band artifact"),
    "strict_mean": mean_strict, "strict_n": n_strict,
    "pooled_p": p_pooled,
    "note": "判据阈值 +0.03 与 dev 噪声带 (±0.030) 同量级；严格复验(同数据 4 seed) 与"
            "跨设定合并(6 组配对) 给出不同强度，两个都要报告。",
}

# ---------- plots ----------
os.makedirs(OUT, exist_ok=True)
curves = {}
for tag, res in [("10% seed1", load("stage9_seed1")), ("boundary", bnd),
                 ("1.5B", big), ("30%", d30), ("10% seed42", ref)]:
    if not res:
        continue
    for g, r in res.items():
        curves[f"{tag}/{g}"] = r["history"]
if curves:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, h in curves.items():
        ax.plot([e["epoch"] for e in h], [e["dev_acc"] for e in h], "o-", lw=1.2, label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("RTE dev acc")
    ax.set_title(f"Stage9 dev curves (dev n={DEV_N}, ±{NOISE:.2f} noise band)")
    ax.axhspan(0.5 - NOISE, 0.5 + NOISE, color="grey", alpha=0.15)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/curves.png", dpi=150)

if stat_rows:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(stat_rows))[::-1]
    ax.errorbar([r["delta_best"] for r in stat_rows], y,
                xerr=None, fmt="o", color="steelblue", ms=7)
    for yi, r in zip(y, stat_rows):
        ax.plot([r["delta_best"], r["delta_last3"]], [yi, yi], color="lightsteelblue", lw=2)
    m = checks["5_paired"]["pooled_all_settings"]["delta_best"]["mean"]
    lo, hi = checks["5_paired"]["pooled_all_settings"]["delta_best"]["ci95"]
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(0.03, color="r", ls="--", lw=1, label="criterion +0.03")
    ax.axvspan(lo, hi, color="green", alpha=0.12, label=f"pooled mean 95%CI")
    ax.axvline(m, color="green", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['setting']} ({r['group']})" for r in stat_rows], fontsize=7)
    ax.set_xlabel("Δ dev acc  (● best-of-5, ─ mean of last 3 epochs)")
    ax.set_title("TAIL6/TAIL7 − baseline, paired by setting"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/deltas.png", dpi=150)

# ---------- markdown ----------
hdr = "| 检验 | 数据 | 组 | dev acc | Δ vs baseline | 切除情况 |\n|---|---|---|---|---|---|"
body = "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
with open(f"{OUT}/stage9_table.md", "w") as f:
    f.write(hdr + "\n" + body + "\n")

print(hdr)
print(body)
print("\n=== paired stats ===")
for r in stat_rows:
    print(f"  {r['setting']:18s} {r['group']:7s} best {r['delta_best']:+.3f}  "
          f"last3 {r['delta_last3']:+.3f}  devloss {r['delta_devloss']:+.3f}")
print(json.dumps({k: checks["5_paired"][k] for k in ("strict_same_data_4seeds", "pooled_all_settings")},
                 indent=2))
print("verdict:", json.dumps(checks["verdict"], ensure_ascii=False, indent=2))

print("\n=== verdicts ===")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                  for k, v in checks.items()}, indent=2, ensure_ascii=False))
if "1_multi_seed" in checks:
    print("per-seed deltas:", {k: round(v["delta"], 3) for k, v in checks["1_multi_seed"]["per_seed"].items()})
    print("note: seed42 reference comes from stage3_qwen/stage5 (same hyper-params, "
          "data subset random_state=42 in the older script) — treat as indicative, "
          "the 3 fresh seeds are the actual replication.")
with open(f"{OUT}/summary.json", "w") as f:
    json.dump({"checks": checks, "rows": rows, "noise_sd": NOISE}, f, indent=2, ensure_ascii=False)
print(f"\nwrote {OUT}/summary.json, stage9_table.md, curves.png")
