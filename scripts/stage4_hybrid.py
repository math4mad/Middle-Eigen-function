"""Stage 4: HYBRID models via cross-model spectral splice.

For matched weight matrices of two same-shape models (BERT-family MLP output
projections), build a brand-new matrix:
  W_hyb = (U_X S_X mask_top) V_X^T  +  (U_Y S_Y mask_mid) V_Y^T
"top 骨架段来自 X + middle 松散段来自 Y"。所有非目标权重与分类头来自 X，
文本用 X 自己的 tokenizer 编码（同 249 行样本、同 seed）。

Partner pairs (all fp32, same vocab/dims):
  BERT(12 layers) <-> DistilBERT(6 layers), distil layer i == bert layer 2i
  BERT(12)        <-> RoBERTa(12), identity mapping (cross tokenizer/vocab!
                     treated as secondary evidence: matrices splice cleanly,
                     but "same index, same meaning" is weaker)

Groups (X backbone first):
  D_*: BERT/DistilBERT family   H_*: BERT/RoBERTa
  e.g. D_Btop_Lmid = top(BERT)+mid(DistilBERT), classifier from BERT.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3_ablate_train as S3  # noqa: E402

P = os.path.abspath
MODELS = {
    "B": P("models/AI-ModelScope/bert-base-uncased"),
    "L": P("models/models/AI-ModelScope--distilbert-base-uncased/snapshots/master"),
    "R": P("models/models/AI-ModelScope--roberta-base/snapshots/master"),
}
SUBSTR = {"B": "output.dense", "L": "ffn.lin2", "R": "output.dense"}
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage4_hybrid")

GROUPS = {  # name -> (X_top_src, Y_mid_src)
    "D_Btop_Lmid": ("B", "L"),
    "D_Ltop_Bmid": ("L", "B"),
    "D_Btop_Bmid": ("B", "B"),
    "D_Ltop_Lmid": ("L", "L"),
    "H_Rtop_Bmid": ("R", "B"),
    "H_Rtop_Rmid": ("R", "R"),
    "H_Btop_Bmid": ("B", "B"),  # same as D_Btop_Bmid, kept for file compat
}


def targets_of(model, key):
    out = []
    for name, mod in model.named_modules():
        if SUBSTR[key] in name and "attention" not in name \
           and "embeddings" not in name and isinstance(mod, nn.Linear):
            out.append((name, mod))
    return out


def pair_maps(n_x, n_y):
    """Y-index for each X-position."""
    if n_x == n_y:
        return list(range(n_y))
    if n_x == 12 and n_y == 6:      # bert <- distil
        return [i // 2 for i in range(12)]
    if n_x == 6 and n_y == 12:      # distil <- bert
        return [2 * i for i in range(6)]
    raise ValueError(f"cannot map {n_x} layers to {n_y}")


def bands(n, kind):
    k_top = max(1, int(np.ceil(n * 0.10)))
    k_bot = max(1, int(np.ceil(n * 0.10)))
    m = np.zeros(n, dtype=bool)
    if kind == "top":
        m[:k_top] = True
    elif kind == "mid":
        m[k_top:n - k_bot] = True
    return m


def band_part(W, kind):
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    m = bands(len(s), kind)
    return (U * (s * m)) @ Vt, s[m], Vt[m]


def build_hybrid(xkey, ykey, report):
    model = AutoModelForSequenceClassification.from_pretrained(MODELS[xkey], num_labels=2)
    ymodel = AutoModelForSequenceClassification.from_pretrained(MODELS[ykey], num_labels=2)
    y_tgt = targets_of(ymodel, ykey)
    x_tgt = targets_of(model, xkey)
    mapping = pair_maps(len(x_tgt), len(y_tgt))
    with torch.no_grad():
        for (name, mod), yi in zip(x_tgt, mapping):
            yname, ymod = y_tgt[yi]
            Wx = mod.weight.detach().float().numpy()
            Wy = ymod.weight.detach().float().numpy()
            if Wx.shape != Wy.shape:
                raise RuntimeError(f"shape mismatch {name} vs {yname}")
            top, s_top, V_top = band_part(Wx, "top")
            mid, s_mid, V_mid = band_part(Wy, "mid")
            mod.weight.copy_(torch.from_numpy(top + mid).to(mod.weight.dtype))
            ov = np.linalg.svd(V_top @ V_mid.T, compute_uv=False)   # [k_top, k_mid]
            report[name] = {"paired_y": yname,
                            "energy_top_from_X": float((s_top ** 2).sum() / (Wx ** 2).sum()),
                            "energy_mid_from_Y": float((s_mid ** 2).sum() / (Wy ** 2).sum()),
                            "fro_rel_change_vs_X": float(np.linalg.norm(Wx - (top + mid)) / np.linalg.norm(Wx)),
                            "overlap_topX_midY": float(ov.mean())}
    return model


def train_hybrid(group, xkey, ykey, device):
    S3.set_all_seeds()
    tok = AutoTokenizer.from_pretrained(MODELS[xkey])
    train_d, dev_d, cols = S3.load_rte(tok)   # same 249 rows (same seed), X tokenizer
    rep = {}
    model = build_hybrid(xkey, ykey, rep)
    w0 = {n: m.weight.detach().float().cpu().numpy().copy()
          for (n, m) in targets_of(model, xkey)}
    model = model.to(device)
    tl = DataLoader(train_d, batch_size=S3.BATCH, shuffle=True)
    dl_ = DataLoader(dev_d, batch_size=S3.BATCH)
    optim = torch.optim.AdamW(model.parameters(), lr=S3.LR, weight_decay=0.01)
    hist = []
    for ep in range(S3.EPOCHS):
        model.train()
        for b in tl:
            b = {c: t.to(device) for c, t in zip(cols + ["labels"], b)}
            y = b.pop("labels")
            loss = nn.functional.cross_entropy(model(**b).logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); optim.zero_grad()
        trl, tra = S3.evaluate(model, tl, device, cols)
        dvl, dva = S3.evaluate(model, dl_, device, cols)
        hist.append({"epoch": ep + 1, "train_loss": trl, "train_acc": tra,
                     "dev_loss": dvl, "dev_acc": dva})
        print(f"[{group}] ep{ep+1} train_loss={trl:.3f} train_acc={tra:.3f} "
              f"dev_loss={dvl:.3f} dev_acc={dva:.3f}", flush=True)
    inv = {n: S3.invasions(w0[n], m.weight.detach().float().cpu().numpy())
           for (n, m) in targets_of(model, xkey)}
    best = max(hist, key=lambda h: h["dev_acc"])
    return {"group": group, "x": xkey, "y": ykey, "splice": rep, "history": hist,
            "best": best, "final": hist[-1], "invasions": inv}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {}
    rf = f"{OUT_DIR}/results.json"
    if os.path.exists(rf):
        results = json.load(open(rf))
    for g in (sys.argv[1:] or list(GROUPS)):
        if g in results:
            print(f"skip {g} (already present)")
            continue
        print(f"\n===== {g} =====", flush=True)
        xk, yk = GROUPS[g]
        try:
            results[g] = train_hybrid(g, xk, yk, device)
        except (RuntimeError, torch.AcceleratorError) as e:   # flaky MPS: retry once
            print(f"[{g}] crashed ({repr(e)[:120]}); retrying", flush=True)
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
            results[g] = train_hybrid(g, xk, yk, device)
        with open(rf, "w") as f:
            json.dump(results, f, indent=2)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for g, r in results.items():
        ax.plot([h["epoch"] for h in r["history"]], [h["dev_acc"] for h in r["history"]],
                "o-", ms=3, label=g)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_title("Hybrid spectral splice, RTE dev (10% train)")
    ax.set_xlabel("epoch"); ax.set_ylabel("dev acc"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/hybrid_curves.png", dpi=150)
    print("\nsummary:")
    for g, r in results.items():
        ov = np.mean([v["overlap_topX_midY"] for v in r["splice"].values()])
        print(f"  {g:12s} dev={r['best']['dev_acc']:.3f}(ep{r['best']['epoch']}) "
              f"tr@best={r['best']['train_acc']:.3f} overlap(topX,midY)={ov:.2f}")
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
