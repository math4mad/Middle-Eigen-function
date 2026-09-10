"""Stage 10 summarizer — does an online spectral monitor track the junk-injection dose?

Reads outputs/stage10/{results.json, monitor_*.jsonl} produced by stage10_inject_monitor.py
and answers the three stage-10 questions:

  Q1 下游损伤：dev acc 是否随注入比例 f 单调下降（剂量-反应）？
  Q2 在线可检测性：哪个谱指标对 f 单调（Spearman）、效应量最大、多早能报警
     （以同 seed 的 f=0 轨迹为零假设，|z|>3 连续 3 次 → 报警，记录报警时刻占训练的比例）？
  Q3 逐层消减能否回收：TAIL6（stage9 的深层 top-10% 切除）在 f=0/0.25/0.5 下的回收量
     是否随 f 增大（"截取的数量与注入比例是否相关"）；M12 作为"切错地方"的对照。

Writes outputs/stage10/{summary.json, stage10_table.md, dose.png, traces.png}.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.environ.get("OUT_ROOT", "outputs")
OUT = f"{ROOT}/stage10"
rp = f"{OUT}/results_partial.json"
R = json.load(open(f"{OUT}/results.json" if os.path.exists(f"{OUT}/results.json") else rp))
METRICS = ["rel", "r_eff", "top_E", "mid_E", "tail_E", "cos_max", "angle"]

runs = []
for tag, r in R.items():
    f, cut, seed = r["junk_frac"], r["cut"], r["seed"]
    mon = [json.loads(l) for l in open(f"{OUT}/monitor_{tag}.jsonl")][1:]
    tailn = max(1, int(0.25 * len(mon)))
    end = {k: float(np.mean([m[k] for m in mon[-tailn:]])) for k in METRICS}
    inv = list(r["invasions"].values())
    dec = np.mean([v["energy_by_decile_of_original_spectrum"] for v in inv], axis=0)
    runs.append({"tag": tag, "f": f, "cut": cut, "seed": seed,
                 "dev": r["best"]["dev_acc"], "dev_last": r["final"]["dev_acc"],
                 "train": r["final"]["train_acc"], "dev_loss": r["final"]["dev_loss"],
                 "n_junk": r["n_junk"], "n_mix": r["n_clean"] + r["n_junk"],
                 "end": end, "deciles": dec, "mon": mon, "hist": r["history"]})

base = [x for x in runs if x["cut"] == "none"]
fs = sorted({x["f"] for x in base})

# ---------- Q1 downstream damage ----------
q1 = []
for f in fs:
    sub = [x for x in base if x["f"] == f]
    q1.append({"f": f, "n_junk": int(np.mean([x["n_junk"] for x in sub])),
               "dev_mean": float(np.mean([x["dev"] for x in sub])),
               "dev_se": float(np.std([x["dev"] for x in sub], ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0,
               "train": float(np.mean([x["train"] for x in sub])),
               "dev_loss": float(np.mean([x["dev_loss"] for x in sub])),
               "per_seed": {str(x["seed"]): round(x["dev"], 3) for x in sub}})
rho_dev = stats.spearmanr([x["f"] for x in base], [-x["dev"] for x in base])

# ---------- Q2 monitor sensitivity ----------
sens = []
for k in METRICS + ["dec_tail", "dec_top"]:
    def val(x):
        if k == "dec_tail":
            return float(np.sum(x["deciles"][5:]))
        if k == "dec_top":
            return float(x["deciles"][0])
        return x["end"][k]
    v = np.array([val(x) for x in base])
    fr = np.array([x["f"] for x in base])
    rho = stats.spearmanr(fr, v)
    v0, v5 = v[fr == 0], v[fr == (max(fs[1:]) if len(fs) > 1 else fs[0])]
    pooled = np.sqrt((v0.var(ddof=1) + v5.var(ddof=1)) / 2) if len(v0) > 1 and len(v5) > 1 else np.nan
    if len(v0) == 0 or len(v5) == 0:
        continue
    sens.append({"metric": k, "spearman_rho_vs_f": float(rho.statistic), "p": float(rho.pvalue),
                 "mean_f0": float(v0.mean()), "mean_f_max": float(v5.mean()),
                 "cohens_d": float((v5.mean() - v0.mean()) / pooled) if pooled and pooled > 0 else np.nan,
                 "by_f": {str(f): float(np.round(v[fr == f].mean(), 4)) for f in fs}})
sens.sort(key=lambda d: -abs(d["spearman_rho_vs_f"]))
if len(fs) < 2:
    print("!! 只有单一 f 完成，Q2/Q3 暂时无法计算（等其余组跑完再重跑本脚本）")

# alarm: |z| > 3 vs the same-seed f=0 trajectory, 3 consecutive samples
def alarm_pos(mon, ref_mon, k):
    ref = np.array([m[k] for m in ref_mon])
    mu, sd = ref.mean(), ref.std(ddof=1)
    if sd <= 0:
        return None
    streak, hit = 0, None
    for i, m in enumerate(mon):
        if abs(m[k] - mu) > 3 * sd:
            streak += 1
            if streak >= 3:
                hit = i / len(mon)
                break
        else:
            streak = 0
    return hit

alarms = {}
for k in [s["metric"] for s in sens if len(fs) > 1 and s["metric"] in METRICS][:4]:
    hits = []
    for seed in sorted({x["seed"] for x in base}):
        ref = [x for x in base if x["seed"] == seed and x["f"] == 0]
        if not ref:
            continue
        ref = ref[0]["mon"]
        for x in [y for y in base if y["seed"] == seed and y["f"] > 0]:
            h = alarm_pos(x["mon"], ref, k)
            if h is not None:
                hits.append({"f": x["f"], "seed": seed, "alarm_at_frac_of_train": round(h, 2)})
    alarms[k] = {"n_alarmed": len(hits), "hits": hits}

# ---------- Q3 cut recovery ----------
q3 = []
for cut in sorted({x["cut"] for x in runs if x["cut"] != "none"}):
    for f in sorted({x["f"] for x in runs if x["cut"] == cut}):
        c = [x for x in runs if x["cut"] == cut and x["f"] == f]
        n = [x for x in base if x["f"] == f and x["seed"] in [y["seed"] for y in c]]
        if not c or not n:
            continue
        q3.append({"cut": cut, "f": f, "seeds": [x["seed"] for x in c],
                   "dev_none": float(np.mean([x["dev"] for x in n])),
                   "dev_cut": float(np.mean([x["dev"] for x in c])),
                   "recovery": float(np.mean([x["dev"] for x in c]) - np.mean([x["dev"] for x in n])),
                   "damage": float(np.mean([x["dev"] for x in n]) -
                                   np.mean([y["dev"] for y in base if y["f"] == 0 and y["seed"] in [z["seed"] for z in c]]))})

# ---------- plots ----------
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
best_metric = sens[0]["metric"]
for ax, k in zip(axes[0], METRICS[:3]):
    for f in fs:
        xs = [x for x in base if x["f"] == f]
        for x in xs:
            m = np.array([s[k] for s in x["mon"]])
            ax.plot(np.arange(len(m)) / max(len(m) - 1, 1), m, lw=1.2,
                    label=f"f={f:.2f} s{x['seed']}", alpha=0.8)
    ax.set_title(f"{k} vs training progress"); ax.set_xlabel("frac of training"); ax.grid(alpha=0.3)
for ax, k in zip(axes[1], ["tail_E", "r_eff", "dev"]):
    for f in fs:
        for x in [y for y in base if y["f"] == f]:
            if k == "dev":
                ax.plot([h["epoch"] for h in x["hist"]], [h["dev_acc"] for h in x["hist"]], "o-",
                        lw=1.2, ms=3, label=f"f={f:.2f} s{x['seed']}")
            else:
                m = np.array([s[k] for s in x["mon"]])
                ax.plot(np.arange(len(m)) / max(len(m) - 1, 1), m, lw=1.2,
                        label=f"f={f:.2f} s{x['seed']}", alpha=0.8)
    ax.set_title(f"{k} vs training"); ax.grid(alpha=0.3)
axes[1][2].set_xlabel("epoch"); axes[1][2].set_ylabel("dev acc")
axes[0][0].legend(fontsize=5, ncol=2)
fig.suptitle("Stage10 · junk-injection dose-response with online ΔW spectral monitors")
fig.tight_layout(); fig.savefig(f"{OUT}/traces.png", dpi=150)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].errorbar([q["f"] for q in q1], [q["dev_mean"] for q in q1],
               yerr=[q["dev_se"] for q in q1], fmt="o-", color="firebrick")
ax[0].set_xlabel("junk fraction f"); ax[0].set_ylabel("best dev acc")
ax[0].set_title(f"Q1 damage (ρ={rho_dev.statistic:+.2f}, p={rho_dev.pvalue:.2f})"); ax[0].grid(alpha=0.3)
for i, k in enumerate(["tail_E", "r_eff"]):
    s = next(x for x in sens if x["metric"] == k)
    ax[i + 1].plot(fs, [s["by_f"][str(f)] for f in fs], "s-", color="teal")
    ax[i + 1].set_title(f"Q2 {k}: ρ={s['spearman_rho_vs_f']:+.2f} d={s['cohens_d']:+.1f}")
    ax[i + 1].set_xlabel("junk fraction f"); ax[i + 1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/dose.png", dpi=150)

# decile profile of the LoRA delta (where the junk writes itself)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for f in fs:
    prof = np.mean([x["deciles"] for x in base if x["f"] == f], axis=0)
    ax.plot(np.arange(1, 11), prof, "o-", lw=1.4, ms=4, label=f"f={f:.2f}")
ax.axvspan(5.5, 10.5, color="grey", alpha=0.12)
ax.text(8, ax.get_ylim()[1] * 0.05 + 0.02, "loose spectrum / tail half", fontsize=8, ha="center")
ax.set_xlabel("decile of the original spectrum (1 = largest σ)"); ax.set_ylabel("energy of ΔW directions")
ax.set_title("Where the invading directions land on the original spectrum"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/invasion_deciles.png", dpi=150)

# ---------- report ----------
lines = []
lines.append("## Q1 下游损伤（剂量-反应）\n")
lines.append("| f | 垃圾条数 | best dev | ±se | train acc | final dev loss |\n|---|---|---|---|---|---|")
for q in q1:
    lines.append(f"| {q['f']:.2f} | {q['n_junk']} | {q['dev_mean']:.3f} | {q['dev_se']:.3f} | "
                 f"{q['train']:.3f} | {q['dev_loss']:.2f} |")
lines.append(f"\nSpearman ρ(f, −dev) = **{rho_dev.statistic:+.2f}** (p={rho_dev.pvalue:.2f})\n")
lines.append("## Q2 在线指标对 f 的敏感度（按 |ρ| 排序）\n")
lines.append("| 指标 | ρ vs f | p | f=0 均值 | f=max 均值 | Cohen's d | 报警命中(组) |\n|---|---|---|---|---|---|---|")
for s in sens:
    a = alarms.get(s["metric"], {})
    lines.append(f"| {s['metric']} | {s['spearman_rho_vs_f']:+.2f} | {s['p']:.3f} | {s['mean_f0']:.4f} | "
                 f"{s['mean_f_max']:.4f} | {s['cohens_d']:+.1f} | {a.get('n_alarmed', '—')} |")
lines.append("\n<details><summary>报警明细</summary>\n\n```json")
lines.append(json.dumps(alarms, indent=1))
lines.append("```\n</details>\n")
lines.append("## Q3 逐层消减的回收（cut vs none，同 seed 同 f）\n")
lines.append("| 切除 | f | dev(none) | dev(cut) | 回收 | 该 f 的损伤 |\n|---|---|---|---|---|---|")
for q in q3:
    lines.append(f"| {q['cut']} | {q['f']:.2f} | {q['dev_none']:.3f} | {q['dev_cut']:.3f} | "
                 f"{q['recovery']:+.3f} | {q['damage']:+.3f} |")
md = "\n".join(lines)
open(f"{OUT}/stage10_table.md", "w").write(md + "\n")
print(md)

summary = {"q1_damage": q1, "q1_spearman": [float(rho_dev.statistic), float(rho_dev.pvalue)],
           "q2_sensitivity": sens, "q2_alarms": alarms, "q3_cut_recovery": q3,
           "runs": [{k: v for k, v in x.items() if k not in ("mon", "hist")} |
                    {"end": x["end"], "deciles": list(map(float, x["deciles"]))} for x in runs]}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2, ensure_ascii=False, default=float)
print(f"\nwrote {OUT}/summary.json, stage10_table.md, traces.png, dose.png, invasion_deciles.png")
