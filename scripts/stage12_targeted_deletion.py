"""Stage 12 (Qwen2.5-0.5B): 决定性一刀 —— 定向删除 vs 随机删除 vs 按谱位删除（等能量、等条数）

问题（stage/stage12.md）：如果"有害方向"在谱上近似均匀散布，那么**按谱位切割 = 掷硬币**；
它同时解释了 stage5 的位置残差 +0.015、stage9 的 +0.021(p=0.28)、stage10 的"垃圾写在中带"。
本脚本先**测出**污染方向，再在**同一条污染训练流**上比较三种删法。

流程（同一协议：RTE 10% 干净子集 + 伪标注垃圾流、LoRA r=8、5 epochs）：
  ① 探针：在"干净+垃圾(f)"流上训练一次 LoRA，取每个 mlp.down_proj 的
     ΔW = s·B@A 的 top-k 右奇异方向 V_junk（k=2，对应 stage10 实测 r_eff≈2.3）；
  ② 手术：回到**原始预训练权重**，在 W₀ 的 SVD 基里删掉 k 个原子（rank-1 分量），三组只差"删哪 k 个"：
        TGT  = 与 V_junk 最对齐的那 k 个原子（按污染指纹定位）
        RAND = 与 TGT **同分位带**的随机 k 个原子（取向随机、能量分布可比）
        SPEC = 后半谱（6–10 分位）里 σ 最大的 k 个原子（= 我们现在这种"按谱位切"）
  ③ 重训：从手术后的模型出发，在**同一条**流上再训 LoRA，比较 dev。
  ④ 参照：no-op（不删任何原子）= stage10 的 f>0 结果。

判据：TGT − RAND > +0.03 ⇒ 污染可定位（但坐标是"污染方向"，不是"第几大 σ"）；
     TGT ≈ RAND ≈ SPEC ⇒ 切割=能量/容量效应，"按大小切割"可被"随便切"替代；
     三者皆 ≤0 ⇒ 撤回删除式手术，只做监测。
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import loralib
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, f"{HERE}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s3 = _load("stage3_llm_lora")
s10 = _load("stage10_inject_monitor")

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.abspath(
    "models/models/Qwen--Qwen2.5-0.5B/snapshots/master")
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage12")
FRAC_TRAIN = float(os.environ.get("FRAC_TRAIN", "0.10"))
DATA_SEED = int(os.environ.get("DATA_SEED", "42"))
JUNK_SEED = int(os.environ.get("JUNK_SEED", "999"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1").split(",")]
JUNK_FRACS = [float(x) for x in os.environ.get("JUNK_FRACS", "0.5,0.25").split(",")]
ARMS = os.environ.get("ARMS", "noop,tgt,rand,spec").split(",")
K = int(os.environ.get("K", "2"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH = int(os.environ.get("BATCH", "16"))

for m in (s3, s10):
    m.MODEL_NAME = MODEL_PATH
    m.MODEL_PATH = MODEL_PATH
    m.RTE_DIR = os.environ.get("RTE_DIR", "data/RTE")
    m.FRAC_TRAIN = FRAC_TRAIN
    m.DATA_SEED = DATA_SEED
    m.EPOCHS = EPOCHS
    m.BATCH = BATCH

TARGETS = None          # filled with the down_proj module names


def stream(frac_junk, tok, clean_d, cols):
    n = int(round(len(clean_d) * frac_junk / (1 - frac_junk))) if frac_junk > 0 else 0
    junk, _ = s10.make_junk(tok, n, np.random.default_rng(JUNK_SEED)) if n else (None, None)
    return s10.mix(clean_d, junk, cols)


def build_model(mask_idx=None):
    """Fresh pretrained model; if mask_idx given {layer_name: [atom indices]}, zero those atoms."""
    model = s3.AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=2, dtype=torch.float32)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id or 0
    model.config.problem_type = "single_label_classification"
    if mask_idx:
        removed = {}
        with torch.no_grad():
            for name, mod in model.named_modules():
                if not (name.endswith("mlp.down_proj") and isinstance(mod, nn.Linear)):
                    continue
                idx = mask_idx.get(name)
                if not idx:
                    continue
                W = mod.weight.detach().float().numpy().copy()      # .numpy() aliases!
                U, s, Vt = np.linalg.svd(W, full_matrices=False)
                keep = np.ones(len(s), bool)
                keep[list(idx)] = False
                Wn = (U * (s * keep)) @ Vt
                mod.weight.copy_(torch.from_numpy(Wn).to(mod.weight.dtype))
                removed[name] = {"removed": [int(i) for i in idx],
                                 "rank_frac": [round(float(i / len(s)), 3) for i in idx],
                                 "energy_removed": float((s[list(idx)] ** 2).sum() / (s ** 2).sum())}
        return model, removed
    return model, {}


def train(model, train_d, dev_d, cols, device, seed):
    s3.SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = s3.inject_lora(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    model = model.to(device)
    tl = DataLoader(train_d, batch_size=BATCH, shuffle=True)
    dl_ = DataLoader(dev_d, batch_size=BATCH)
    optim = torch.optim.AdamW(trainable, lr=s3.LR)
    hist = []
    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            b = {c: t.to(device) for c, t in zip(cols + ["labels"], b)}
            y = b.pop("labels")
            loss = nn.functional.cross_entropy(model(**b).logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step(); optim.zero_grad()
        trl, tra = s3.evaluate(model, tl, device, cols)
        dvl, dva = s3.evaluate(model, dl_, device, cols)
        hist.append({"epoch": ep + 1, "train_loss": trl, "train_acc": tra,
                     "dev_loss": dvl, "dev_acc": dva})
        print(f"  ep{ep+1} train_acc={tra:.3f} dev_acc={dva:.3f} dev_loss={dvl:.2f}", flush=True)
    return model, hist, trainable


def probe_axes(frac_junk, seed, tok, clean_d, dev_d, cols, device):
    """Train once on the contaminated stream, return per-layer top-K right-singular
    directions of the merged LoRA delta ΔW (the 'junk axes'), plus the base spectra."""
    model, _ = build_model()
    names = [n for n, m in model.named_modules()
             if n.endswith("mlp.down_proj") and isinstance(m, nn.Linear)]
    base = {n: dict(model.named_modules())[n].weight.detach().float().numpy().copy() for n in names}
    model, hist, _ = train(model, stream(frac_junk, tok, clean_d, cols), dev_d, cols, device, seed)
    axes = {}
    with torch.no_grad():
        for n, m in model.named_modules():
            if n in base and isinstance(m, loralib.Linear):
                dW = s3.lora_delta(m)
                _, _, Vt = np.linalg.svd(dW, full_matrices=False)
                axes[n] = Vt[:K].copy()
    return axes, hist, base


def choose_masks(axes, base, arm, seed, tgt=None):
    """Per layer: pick K atoms of W₀ to zero out, under three different coordinates.
    `tgt` (the TGT pick) is passed in for the band-matched RAND arm so both arms remove
    the same NUMBER of atoms from the same spectral BANDS."""
    rng = np.random.default_rng(31337 + seed)
    masks = {}
    for n, W in base.items():
        _, s, V0t = np.linalg.svd(W, full_matrices=False)
        nn_ = len(s)
        band = lambda i: min(int(i / nn_ * 10), 9)
        if arm == "tgt":
            pick, used = [], set()
            cos = np.abs(axes[n] @ V0t.T)                       # [K, n]
            for r in range(cos.shape[0]):
                for j in np.argsort(-cos[r]):
                    if int(j) not in used:
                        pick.append(int(j)); used.add(int(j)); break
            masks[n] = pick
        elif arm == "rand":
            ref = (tgt or choose_masks(axes, base, "tgt", seed))[n]
            per = nn_ // 10
            masks[n] = [int(rng.integers(band(t) * per, min((band(t) + 1) * per, nn_))) for t in ref]
        elif arm == "spec":
            masks[n] = [int(i) for i in range(nn_ // 2, nn_ // 2 + K)]   # 6–10 分位里 σ 最大的 K 个
        else:
            raise ValueError(arm)
    return masks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, "K =", K, "arms:", ARMS, "fracs:", JUNK_FRACS, flush=True)
    tok = s3.AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    clean_d, dev_d, cols = s3.load_rte(tok)
    p = f"{OUT_DIR}/results_partial.json"
    results = json.load(open(p)) if os.path.exists(p) else {}

    for seed in SEEDS:
        for f in JUNK_FRACS:
            tr = stream(f, tok, clean_d, cols)
            if "noop" in ARMS and f"f{int(f*100)}_noop_s{seed}" not in results:
                print(f"\n===== f={f} arm=noop seed={seed} =====", flush=True)
                m0, _ = build_model()
                _, hist, _ = train(m0, tr, dev_d, cols, device, seed)
                results[f"f{int(f*100)}_noop_s{seed}"] = {"junk_frac": f, "arm": "noop", "seed": seed,
                                                          "history": hist, "best": max(hist, key=lambda h: h["dev_acc"])}
                json.dump(results, open(p, "w"), indent=2)
            key_axes = f"f{int(f*100)}_axes_s{seed}"
            if key_axes not in results:
                print(f"\n===== probe f={f} seed={seed} =====", flush=True)
                axes, hist, base = probe_axes(f, seed, tok, clean_d, dev_d, cols, device)
                np.savez(f"{OUT_DIR}/axes_f{int(f*100)}_s{seed}.npz",
                         **{k.replace(".", "__"): v for k, v in axes.items()})
                results[key_axes] = {"probe_history": hist,
                                     "probe_best": max(hist, key=lambda h: h["dev_acc"])}
                json.dump(results, open(p, "w"), indent=2)
            else:
                z = np.load(f"{OUT_DIR}/axes_f{int(f*100)}_s{seed}.npz")
                axes = {k.replace("__", "."): v for k, v in z.items()}
                m0, _ = build_model()
                nm = dict(m0.named_modules())
                base = {k: nm[k].weight.detach().float().numpy().copy() for k in axes}
            tgt = choose_masks(axes, base, "tgt", seed) if "rand" in ARMS else None
            for arm in [a for a in ARMS if a != "noop"]:
                tag = f"f{int(f*100)}_{arm}_s{seed}"
                if tag in results:
                    continue
                print(f"\n===== f={f} arm={arm} seed={seed} =====", flush=True)
                masks = choose_masks(axes, base, arm, seed, tgt=tgt)
                mm, removed = build_model(masks)
                e = [v["energy_removed"] for v in removed.values()]
                rf = [r for v in removed.values() for r in v["rank_frac"]]
                print(f"  surgery: {len(removed)} layers × {K} atoms, "
                      f"energy_removed/layer mean={np.mean(e):.4f}, "
                      f"谱位分布 mean={np.mean(rf):.2f} (0=最大σ, 1=最小σ)", flush=True)
                _, hist, _ = train(mm, tr, dev_d, cols, device, seed)
                results[tag] = {"junk_frac": f, "arm": arm, "seed": seed, "history": hist,
                                "best": max(hist, key=lambda h: h["dev_acc"]),
                                "energy_removed_mean": float(np.mean(e)),
                                "removed_rank_frac": [round(float(x), 3) for x in sorted(rf)]}
                json.dump(results, open(p, "w"), indent=2)
    json.dump(results, open(f"{OUT_DIR}/results.json", "w"), indent=2)
    print("\nsummary:")
    for tag, r in sorted(results.items()):
        if "best" in r:
            print(f"  f={r['junk_frac']:.2f} {r['arm']:5s} s{r['seed']}  dev={r['best']['dev_acc']:.3f}"
                  f"  E_removed={r.get('energy_removed_mean', 0):.4f}")


if __name__ == "__main__":
    sys.exit(main())
