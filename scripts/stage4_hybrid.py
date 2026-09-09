"""Stage 4: HYBRID models via cross-model spectral splice.

For every ablated weight matrix shape (e.g. mlp c_proj 768x3072), take
  - the TOP-10% singular directions+values from model X, and
  - the MIDDLE-80% singular directions+values from model Y,
and sum them into a brand-new matrix:
  W_hyb = (Ux Sx mask_top) Vx^T  +  (Uy Sy mask_mid) Vy^T
All non-target weights (embeddings, attention, LN...) come from X.

Groups:
  H_Btop_Rmid : top bert     + mid roberta
  H_Rtop_Bmid : top roberta  + mid bert
  H_Btop_Bmid : top bert     + mid bert      (same-source splice control)
  H_Rtop_Rmid : top roberta  + mid roberta   (same-source splice control)

Plus a diagnostic: subspace overlap (mean cos of principal angles) between
X's and Y's top / middle right-singular subspaces per layer.

Evaluated exactly like stage3: 10% RTE train (same 249 samples, seed 42),
identical hyper-params, RTE dev.
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
from transformers import AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3_ablate_train as S3  # noqa: E402

BERT = os.environ.get("MODEL_PATH_BERT") or os.path.abspath("models/AI-ModelScope/bert-base-uncased")
ROBERTA = os.environ.get("MODEL_PATH_ROBERTA") or os.path.abspath("models/models/AI-ModelScope--roberta-base/snapshots/master")
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage4_hybrid")

GROUPS = {
    "H_Btop_Rmid": (BERT, ROBERTA),
    "H_Rtop_Bmid": (ROBERTA, BERT),
    "H_Btop_Bmid": (BERT, BERT),
    "H_Rtop_Rmid": (ROBERTA, ROBERTA),
}


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


def build_hybrid(path_top, path_mid, report):
    model = AutoModelForSequenceClassification.from_pretrained(path_top, num_labels=2)
    _mid = AutoModelForSequenceClassification.from_pretrained(path_mid, num_labels=2)
    # index partner state by prefix-stripped suffix (bert./roberta. prefixes differ)
    mid_state = {k.split(".", 1)[1]: v for k, v in _mid.state_dict().items()}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not S3.is_mlp_cproj(name):
                continue
            Wt = p.detach().float().numpy()
            suffix = name.split(".", 1)[1]      # drop this model's prefix too
            if suffix not in mid_state:
                raise RuntimeError(f"no matching matrix in partner model: {name}")
            Wm = mid_state[suffix].detach().float().numpy()
            if Wt.shape != Wm.shape:
                raise RuntimeError(f"shape mismatch {name} {Wt.shape} vs {Wm.shape}")
            top, s_top, V_top = band_part(Wt, "top")
            mid, s_mid, V_mid = band_part(Wm, "mid")
            p.copy_(torch.from_numpy(top + mid).to(p.dtype))
            # subspace overlap: mean cos of principal angles between right-subspaces
            Qa, Qb = V_top.T, V_mid.T           # [r, n] orthonormal columns
            ov = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
            report[name] = {
                "energy_top_from_X": float((s_top ** 2).sum() / (np.linalg.norm(Wt) ** 2)),
                "energy_mid_from_Y": float((s_mid ** 2).sum() / (np.linalg.norm(Wm) ** 2)),
                "fro_rel_change": float(np.linalg.norm(Wt - (top + mid)) / np.linalg.norm(Wt)),
                "subspace_overlap_topX_midY": float(ov.mean()),
            }
    return model


def train_hybrid(group, path_top, path_mid, train_d, dev_d, cols, device):
    S3.set_all_seeds()
    rep = {}
    model = build_hybrid(path_top, path_mid, rep)
    w0 = {n: p.detach().float().cpu().numpy().copy()
          for n, p in model.named_parameters() if S3.is_mlp_cproj(n)}
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
    inv = {n: S3.invasions(w0[n], p.detach().float().cpu().numpy())
           for n, p in model.named_parameters() if S3.is_mlp_cproj(n)}
    best = max(hist, key=lambda h: h["dev_acc"])
    return {"group": group, "splice": rep, "history": hist, "best": best,
            "final": hist[-1], "invasions": inv}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = S3.AutoTokenizer.from_pretrained(BERT)
    train_d, dev_d, cols = S3.load_rte(tok)
    groups = sys.argv[1:] or list(GROUPS)
    results = {}
    for g in groups:
        print(f"\n===== {g} =====", flush=True)
        results[g] = train_hybrid(g, *GROUPS[g], train_d, dev_d, cols, device)
        with open(f"{OUT_DIR}/results.json", "w") as f:
            json.dump(results, f, indent=2)
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, r in results.items():
        ax.plot([h["epoch"] for h in r["history"]], [h["dev_acc"] for h in r["history"]],
                "o-", label=g)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_title("Hybrid spectral splice, RTE dev (10% train)")
    ax.set_xlabel("epoch"); ax.set_ylabel("dev acc"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/hybrid_curves.png", dpi=150)
    print("\nsummary:")
    for g, r in results.items():
        ov = np.mean([v["subspace_overlap_topX_midY"] for v in r["splice"].values()])
        print(f"  {g:12s} dev={r['best']['dev_acc']:.3f}(ep{r['best']['epoch']}) "
              f"tr@best={r['best']['train_acc']:.3f} Xtop/Ymid-subspace-overlap={ov:.2f}")
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
