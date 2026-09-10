"""两种排序的一致性检验：magnitude order (σ 从大到小) vs functional order (损伤后 Δloss 从大到小)

用户命题（2026-09-10）：
    "数据科学要求特征值按大小排列，智能科学要求按功能方向排列。"

本脚本把它变成一个数：**同一个矩阵的两套排序，Kendall τ 有多大？**

做法（lesion / 布洛卡式探针，零训练、纯前向）：
  1. 取 Qwen2.5-0.5B 的若干 mlp.down_proj，做 SVD 得到方向 v_i（右奇异向量）与 σ_i；
  2. **单方向损伤**：W ← W − σ_i u_i v_iᵀ（只拿掉这一个 rank-1 分量），
     在 RTE dev 文本上量语言建模损失的变化 Δloss_i ⇒ 该方向的**功能重要性**；
  3. **分位带损伤**：拿掉第 d 个十分位带里的全部方向 ⇒ "能量顺序 vs 功能顺序" 的宏观曲线；
  4. 输出：每层 τ(σ-rank, importance-rank)、top 带 Δloss / tail 带 Δloss、
     以及"最重要的 8 个方向里有几个落在 σ 最大的前 10%"（= 两套排序重合度的直接读数）。

若 τ≈1：按大小切 = 按功能切（数据科学的排序够用）；
若 τ≈0：σ 的大小与功能重要性**无关** ⇒ "按大小切割"在功能意义上等价于随机切割。
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.abspath(
    "models/models/Qwen--Qwen2.5-0.5B/snapshots/master")
RTE_DIR = os.environ.get("RTE_DIR", "data/RTE")
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage13")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,12,20,23").split(",")]
SUBSTR = os.environ.get("ABLATE_SUBSTR", "mlp.down_proj")
N_BAND = int(os.environ.get("N_BAND", "10"))         # deciles
PER_BAND = int(os.environ.get("PER_BAND", "8"))      # single-direction lesions per band
NSAMPLE = int(os.environ.get("NSAMPLE", "128"))       # dev examples used
BATCH = int(os.environ.get("BATCH", "16"))
SEED = int(os.environ.get("SEED", "0"))


def load_text(tok):
    import pandas as pd
    df = pd.read_csv(f"{RTE_DIR}/dev.tsv", sep="\t").sample(NSAMPLE, random_state=SEED)
    txt = [f"{a} {b}" for a, b in zip(df["sentence1"], df["sentence2"])]
    enc = tok(txt, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
    return enc


@torch.no_grad()
def ce_loss(model, enc, device):
    """Causal-LM loss on REAL tokens only; padding positions are masked with -100.
    (Without this the number is dominated by "predict the pad token" and every lesion
    looks like it *lowers* the loss — an earlier run showed exactly that artifact.)"""
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    labels = ids.clone()
    labels[am == 0] = -100
    tot, n = 0.0, 0
    for i in range(0, ids.shape[0], BATCH):
        x, y = ids[i:i + BATCH], labels[i:i + BATCH]
        out = model(input_ids=x, attention_mask=am[i:i + BATCH], labels=y)
        k = int((y != -100).sum().item())
        tot += out.loss.item() * k
        n += k
    return tot / n


def lesion(W, U, s, Vt, idxs):
    Wn = W.copy()
    for i in idxs:
        Wn -= np.outer(U[:, i] * s[i], Vt[i])
    return Wn


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).to(device)
    model.eval()
    enc = load_text(tok)
    targets = {f"model.layers.{i}.{SUBSTR}.weight": i for i in LAYERS}
    rng = np.random.default_rng(SEED)
    report, t0 = {}, time.time()
    base = ce_loss(model, enc, device)
    ntok = int((enc["attention_mask"] == 1).sum())
    print(f"baseline causal-LM loss on RTE dev({NSAMPLE} ex / {ntok} real tokens) = {base:.4f}", flush=True)

    for name, li in targets.items():
        mod = model.get_submodule(name[:-len(".weight")])
        W0 = mod.weight.detach().float().cpu().numpy().copy()   # .numpy() aliases the tensor!
        U, s, Vt = np.linalg.svd(W0, full_matrices=False)
        n = len(s)
        edges = np.linspace(0, n, N_BAND + 1).astype(int)
        entry = {"layer": li, "shape": list(W0.shape), "band": [], "single": []}
        # --- band lesions (drop the whole decile band) ---
        for b in range(N_BAND):
            idxs = list(range(edges[b], edges[b + 1]))
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(lesion(W0, U, s, Vt, idxs)).to(mod.weight.dtype))
            entry["band"].append({"band": b + 1, "sigma_range": [float(s[idxs[-1]]), float(s[idxs[0]])],
                                  "energy_share": float((s[idxs] ** 2).sum() / (s ** 2).sum()),
                                  "dloss": ce_loss(model, enc, device) - base})
        # --- single-direction lesions: PER_BAND samples from each band ---
        picks = []
        for b in range(N_BAND):
            idxs = list(range(edges[b], edges[b + 1]))
            picks += [int(i) for i in rng.choice(idxs, size=min(PER_BAND, len(idxs)), replace=False)]
        for i in picks:
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(lesion(W0, U, s, Vt, [i])).to(mod.weight.dtype))
            entry["single"].append({"idx": i, "sigma_rank_frac": float(i / n),
                                    "sigma": float(s[i]), "dloss": ce_loss(model, enc, device) - base})
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W0).to(mod.weight.dtype))     # restore
        from scipy import stats
        sr = [x["sigma_rank_frac"] for x in entry["single"]]
        im = [x["dloss"] for x in entry["single"]]
        tau, p = stats.kendalltau(sr, im)
        top8 = sorted(entry["single"], key=lambda x: -x["dloss"])[:8]
        entry["kendall_tau_sigma_vs_importance"] = float(tau)
        entry["tau_p"] = float(p)
        entry["top8_in_sigma_top10pct"] = int(sum(x["sigma_rank_frac"] < 0.10 for x in top8))
        entry["top8_band_hist"] = [int(sum(1 for x in top8 if 0.10 * b <= x["sigma_rank_frac"] < 0.10 * (b + 1)))
                                   for b in range(N_BAND)]
        report[f"L{li}.{SUBSTR}"] = entry
        print(f"L{li:2d} τ(σrank,importance)={tau:+.2f} (p={p:.1e})  "
              f"top-band Δloss={entry['band'][0]['dloss']:+.4f}  "
              f"tail-band Δloss={entry['band'][-1]['dloss']:+.4f}  "
              f"最重要8方向中落在σ前10%的个数={entry['top8_in_sigma_top10pct']}/8  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    taus = [v["kendall_tau_sigma_vs_importance"] for v in report.values()]
    band_curve = np.mean([[b["dloss"] for b in v["band"]] for v in report.values()], axis=0)
    band_energy = np.mean([[b["energy_share"] for b in v["band"]] for v in report.values()], axis=0)
    out = {"baseline_loss": base, "per_layer": report,
           "mean_kendall_tau": float(np.mean(taus)),
           "mean_band_dloss": [float(x) for x in band_curve],
           "mean_band_energy": [float(x) for x in band_energy],
           "top8_in_top10pct": int(np.mean([v["top8_in_sigma_top10pct"] for v in report.values()]))}
    json.dump(out, open(f"{OUT_DIR}/two_orderings.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax[0].plot(np.arange(1, N_BAND + 1), band_curve, "o-", color="firebrick", label="dloss (functional importance)")
    ax[0].plot(np.arange(1, N_BAND + 1), band_energy * band_curve.max() / max(band_energy.max(), 1e-9),
               "s--", color="grey", lw=1, label="band energy share (rescaled)")
    ax[0].set_xlabel("sigma decile band (1 = largest sigma)"); ax[0].set_ylabel("dloss")
    ax[0].set_title("Band lesion: energy ordering vs functional ordering"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    for li, v in report.items():
        ax[1].scatter([x["sigma_rank_frac"] for x in v["single"]],
                      [x["dloss"] for x in v["single"]], s=12, alpha=0.7, label=li)
    ax[1].set_xlabel("position of the direction in sigma order (0 = largest)"); ax[1].set_ylabel("single-direction lesion dloss")
    ax[1].set_title(f"Single-direction lesions: mean Kendall tau = {out['mean_kendall_tau']:+.2f}")
    ax[1].legend(fontsize=6); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/two_orderings.png", dpi=150)
    print(f"mean Kendall τ = {out['mean_kendall_tau']:+.3f}; "
          f"最重要 8 方向落在 σ 前 10% 的个数 = {out['top8_in_top10pct']}/8")
    print(f"wrote {OUT_DIR}/two_orderings.json, two_orderings.png")


if __name__ == "__main__":
    sys.exit(main())
