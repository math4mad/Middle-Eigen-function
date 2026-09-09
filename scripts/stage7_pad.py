"""Stage 7: cross-SIZE hybrids via matrix padding.

Stage 4 spliced same-shape matrices (BERT<->RoBERTa/DistilBERT, 768x3072).
Here the donor Y may be SMALLER (electra-small: 256x1024 c_proj): embed its
middle-band into the host's matrix space by zero-padding, then splice onto
the host's top band:

    W_hyb = top10%(X)  +  PAD( mid80%(Y) )

Four embedding variants test WHAT could make a foreign middle work:
  pad_tl      : identity placement, Y's band into X's first dims (electra's
                channel convention "aligned" by matrix index only)
  pad_rot     : the padded band is additionally rotated by random orthogonal
                Q (rows) / R (cols) — same spectrum & energy, no index alignment
                (if pad_tl >> pad_rot, the alignment of conventions matters)
  pad_scaled  : pad_tl, but rescaled so its Frobenius^2 equals the mid-band
                fraction the same-position host band would keep (~2.3x top-band
                energy) — rules out "donor energy was too small to matter"
  pad_random  : replace Y by a RANDOM GAUSSIAN matrix with electra's singular
                VALUE profile — pure noise with the right spectrum shape:
                the floor every donor must beat.

Host/backbone X = bert-base (12 c_proj matrices). Dev/train/eval identical
to stage3/4 (same 249-sample subset, lr 2e-5, 8 epochs, dev 277).
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
HOST = P("models/AI-ModelScope/bert-base-uncased")
DONORS = {
    "E": P("models/models/google--electra-small-discriminator/snapshots/master"),  # 256x1024
    "B": HOST,
}
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage7_pad")
SUBSTR = "output.dense"


def is_host_target(name):
    return SUBSTR in name and "attention" not in name and "embeddings" not in name \
        and name.endswith("weight")


def bands(n, kind):
    kt = max(1, int(np.ceil(n * 0.10))); kb = max(1, int(np.ceil(n * 0.10)))
    m = np.zeros(n, bool)
    if kind == "top":
        m[:kt] = True
    elif kind == "mid":
        m[kt:n - kb] = True
    return m


def band_of(W, kind, s_override=None):
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    if s_override is not None:
        s = s_override
    m = bands(len(s), kind)
    return (U * (s * m)) @ Vt, s[m]


def pad_to(A, shape):
    out = np.zeros(shape, dtype=np.float64)
    out[:A.shape[0], :A.shape[1]] = A
    return out


def rand_orth(d, rng):
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q


def build(group, device):
    rng = np.random.default_rng(SEED := 42)
    model = AutoModelForSequenceClassification.from_pretrained(HOST, num_labels=2)
    host_mats = {n: p.detach().float().numpy().copy() for n, p in model.named_parameters() if is_host_target(n)}
    if group == "pad_random":
        donor = "E"; donor_state = None
    else:
        donor = "E"
    dm = AutoModelForSequenceClassification.from_pretrained(DONORS[donor], num_labels=2)
    dstate = {n.split(".", 1)[1]: p.detach().float().numpy()
              for n, p in dm.named_parameters()
              if SUBSTR in n and "attention" not in n and "embeddings" not in n and n.endswith("weight")}
    dkeys = sorted(dstate, key=lambda k: int(k.split(".")[2]))
    report = {}
    with torch.no_grad():
        for i, (name, p) in enumerate([pp for pp in model.named_parameters() if is_host_target(pp[0])]):
            Wx = host_mats[name]
            top, s_top = band_of(Wx, "top")
            Wd = dstate[dkeys[i]]
            if Wd.shape[0] >= Wx.shape[0]:
                raise ValueError("donor must be smaller")
            if group == "pad_random":
                # random directions, donor's singular-VALUE profile preserved
                sd = np.linalg.svd(Wd, compute_uv=False)
                U0 = np.linalg.qr(rng.standard_normal(Wd.shape))[0]
                V0 = np.linalg.qr(rng.standard_normal((Wd.shape[1], Wd.shape[0])))[0][:, : Wd.shape[0]]
                Wd = (U0 * sd) @ V0.T
            mid_d, s_mid = band_of(Wd, "mid")
            M = pad_to(mid_d, Wx.shape)
            if group == "pad_rot":
                Q = rand_orth(Wx.shape[0], rng); R = rand_orth(Wx.shape[1], rng)
                M = Q @ M @ R.T
            if group == "pad_scaled":
                target_ratio = 0.685 / 0.295            # host mid/top energy ratio
                cur = (M ** 2).sum() / (top ** 2).sum()
                M *= np.sqrt(target_ratio / cur)
            W = top + M
            p.copy_(torch.from_numpy(W).to(p.dtype))
            e_top = float((top ** 2).sum() / (Wx ** 2).sum())
            report[name] = {"donor_shape": list(Wd.shape), "energy_top_X": e_top,
                            "energy_midY_padded": float((M ** 2).sum() / (Wx ** 2).sum()),
                            "fro_rel_change": float(np.linalg.norm(Wx - W) / np.linalg.norm(Wx))}
    return model, report


def train_group(group, device):
    S3.set_all_seeds()
    model, rep = build(group, device)
    w0 = {n: p.detach().float().cpu().numpy().copy() for n, p in model.named_parameters() if is_host_target(n)}
    tok = AutoTokenizer.from_pretrained(HOST)
    train_d, dev_d, cols = S3.load_rte(tok)
    tl = DataLoader(train_d, batch_size=S3.BATCH, shuffle=True)
    dl_ = DataLoader(dev_d, batch_size=S3.BATCH)
    model = model.to(device)
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
           for n, p in model.named_parameters() if is_host_target(n)}
    best = max(hist, key=lambda h: h["dev_acc"])
    return {"group": group, "pad": rep, "history": hist, "best": best,
            "final": hist[-1], "invasions": inv}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rf = f"{OUT_DIR}/results.json"
    results = json.load(open(rf)) if os.path.exists(rf) else {}
    for g in (sys.argv[1:] or ["pad_tl", "pad_rot", "pad_scaled", "pad_random"]):
        if g in results:
            print("skip", g); continue
        print(f"\n===== {g} =====", flush=True)
        results[g] = train_group(g, device)
        with open(rf, "w") as f:
            json.dump(results, f, indent=2)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for g, r in results.items():
        ax.plot([h["epoch"] for h in r["history"]], [h["dev_acc"] for h in r["history"]], "o-", ms=3, label=g)
    for g, lab, col in [("baseline", "BERT baseline", "#2c3e50"), ("D_Btop_Lmid", "same-size hybrid (distil)", "#27ae60")]:
        src = json.load(open("outputs/stage3/results.json" if g == "baseline" else "outputs/stage4_hybrid/results.json"))
        r = src["baseline"] if g == "baseline" else src[g]
        ax.plot([h["epoch"] for h in r["history"]], [h["dev_acc"] for h in r["history"]],
                "--", color=col, lw=1, label=lab)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_title("Stage7: cross-size hybrids via padding (host=BERT + mid(electra-256))")
    ax.set_xlabel("epoch"); ax.set_ylabel("dev acc"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/pad_curves.png", dpi=150)
    print("\nsummary:")
    for g, r in results.items():
        pr = list(r["pad"].values())[0]
        print(f"  {g:10s} dev={r['best']['dev_acc']:.3f}(ep{r['best']['epoch']}) "
              f"tr@best={r['best']['train_acc']:.3f} midEnergy={pr['energy_midY_padded']:.3f}")
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
