"""Stage 3: three-way SVD ablation + small-data fine-tuning on GLUE/RTE.

Groups (per AGETNTS.md), applied to every mlp.c_proj weight matrix of
bert-base-uncased (12 matrices, shape [768, 3072]):

  A "rigid skeleton": keep the top-10% singular values, zero out the rest.
  B "loose spectrum": drop top-10% and bottom-10%, keep the middle 80%.
  C "random":         keep a random 80% of singular values.
  baseline:           untouched pretrained weights.

All four models are fine-tuned with identical hyper-parameters on a 10%
subsample of RTE train, evaluated on RTE dev. We log train/dev loss+acc,
the generalization gap, and the spectral position of the fine-tuning
"invasion dimensions" (SVD of deltaW = W_finetuned - W_initial, measured
against the original spectrum).
"""
import copy
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset  # noqa: F401 (kept for optional hub loading)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = os.environ.get("MODEL_PATH") or os.path.abspath("models/AI-ModelScope/bert-base-uncased")
RTE_DIR = os.environ.get("RTE_DIR", "data/RTE")
ABLATED_KEY = "output.dense.weight"   # BERT's mlp.c_proj == layer.output.dense (NOT attention.output.dense)


def is_mlp_cproj(name: str) -> bool:
    return name.endswith(ABLATED_KEY) and ".attention." not in name

OUT_DIR = "outputs/stage3"
SEED = 42
BATCH = int(os.environ.get("BATCH", 16))
EPOCHS = int(os.environ.get("EPOCHS", 8))
LR = float(os.environ.get("LR", 2e-5))
FRAC_TRAIN = 0.10


def set_all_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def keep_mask_group(n, group, rng):
    """Boolean mask over the n singular values (sorted descending) to KEEP."""
    k_top = max(1, int(np.ceil(n * 0.10)))
    k_bot = max(1, int(np.ceil(n * 0.10)))
    mask = np.zeros(n, dtype=bool)
    if group == "A":                       # rigid skeleton: top 10%
        mask[:k_top] = True
    elif group == "B":                     # loose spectrum: middle 80%
        mask[k_top:n - k_bot] = True
    elif group == "C":                     # random 80%
        idx = rng.permutation(n)[: int(n * 0.80)]
        mask[idx] = True
    elif group == "baseline":
        mask[:] = True
    else:
        raise ValueError(group)
    return mask


def ablate_model(model, group):
    """Zero selected singular values of each mlp.c_proj and rebuild in place."""
    rng = np.random.default_rng(SEED)
    report = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not is_mlp_cproj(name):
                continue
            W = p.detach().float().numpy()          # [out, in] = [768, 3072]
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            mask = keep_mask_group(len(S), group, rng)
            S2 = S * mask
            p.copy_(torch.from_numpy((U * S2) @ Vt).to(p.dtype))
            report[name] = {
                "kept": int(mask.sum()), "total": int(mask.size),
                "energy_kept": float((S2 ** 2).sum() / (S ** 2).sum()),
                "fro_rel_change": float(np.linalg.norm(S - S2) / np.linalg.norm(S)),
            }
    return report


def invasions(w0, w1):
    """Where does deltaW = w1 - w0 live relative to w0's spectrum?

    Returns, per leading delta right-singular direction, the rank position
    (fraction) in w0's own sorted spectrum of the closest original direction
    (max |cos| over original right singular vectors), plus a histogram over
    deciles of the original spectrum.
    """
    _, s0, V0t = np.linalg.svd(w0, full_matrices=False)
    _, _, Vt = np.linalg.svd(w1 - w0, full_matrices=False)
    k = min(16, Vt.shape[0])          # top-16 delta directions
    cos = np.abs(Vt[:k] @ V0t.T)      # [k, r0] overlap with original directions
    best = cos.max(axis=1)
    rank = cos.argmax(axis=1) / V0t.shape[0]
    weighted_hist, _ = np.histogram(rank, bins=10, range=(0, 1), weights=best / best.sum())
    return {"rank_frac_top_delta_dirs": rank[:8].round(3).tolist(),
            "cos_to_original": best[:8].round(3).tolist(),
            "energy_by_decile_of_original_spectrum": weighted_hist.round(3).tolist(),
            "s0_ref": s0.tolist()}


def load_rte(tokenizer):
    import pandas as pd
    lab = {"not_entailment": 0, "entailment": 1}
    splits = {}
    for split, fn in [("train", "train.tsv"), ("validation", "dev.tsv")]:
        df = pd.read_csv(f"{RTE_DIR}/{fn}", sep="\t")
        df["label"] = df["label"].map(lab)
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)
        splits[split] = df
    print(f"RTE sizes: train={len(splits['train'])} dev={len(splits['validation'])}")
    set_all_seeds()
    train = splits["train"].sample(frac=FRAC_TRAIN, random_state=SEED).reset_index(drop=True)
    print(f"RTE small-data train: {len(train)} samples (10%)")

    def feats(df):
        enc = tokenizer(df["sentence1"].tolist(), df["sentence2"].tolist(), truncation=True,
                        max_length=128, padding="max_length")
        cols = ["input_ids", "attention_mask", "token_type_ids"]
        if "token_type_ids" not in enc:
            cols = ["input_ids", "attention_mask"]
        tens = [torch.tensor(enc[c]) for c in cols] + [torch.tensor(df["label"].tolist())]
        return TensorDataset(*tens), cols
    tr, cols = feats(train)
    dv, _ = feats(splits["validation"])
    return tr, dv, cols


@torch.no_grad()
def evaluate(model, loader, device, cols):
    model.eval()
    lossf = nn.CrossEntropyLoss(reduction="sum")
    tot_l, n, correct = 0.0, 0, 0
    for b in loader:
        b = {c: t.to(device) for c, t in zip(cols + ["labels"], b)}
        out = model(**b)
        tot_l += lossf(out.logits, b["labels"]).item()
        correct += (out.logits.argmax(-1) == b["labels"]).sum().item()
        n += b["labels"].size(0)
    model.train()
    return tot_l / n, correct / n


def train_one(group, train_d, dev_d, cols, device):
    set_all_seeds()
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    ab_report = ablate_model(base, group)
    w0 = {n: p.detach().float().cpu().numpy().copy()
          for n, p in base.named_parameters() if is_mlp_cproj(n)}
    model = base.to(device)
    tl = DataLoader(train_d, batch_size=BATCH, shuffle=True)
    dl_ = DataLoader(dev_d, batch_size=BATCH)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, 0, len(tl) * EPOCHS)
    hist = []
    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            b = {c: t.to(device) for c, t in zip(cols + ["labels"], b)}
            loss = model(**b).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
        trl, tra = evaluate(model, tl, device, cols)
        dvl, dva = evaluate(model, dl_, device, cols)
        hist.append({"epoch": ep + 1, "train_loss": trl, "train_acc": tra,
                     "dev_loss": dvl, "dev_acc": dva, "gap": trl - dvl})
        print(f"[{group}] ep{ep+1} train_loss={trl:.3f} train_acc={tra:.3f} "
              f"dev_loss={dvl:.3f} dev_acc={dva:.3f}", flush=True)
    inv = {n: invasions(w0[n], p.detach().float().cpu().numpy())
           for n, p in model.named_parameters() if is_mlp_cproj(n)}
    best = max(hist, key=lambda h: h["dev_acc"])
    return {"group": group, "ablation": ab_report, "history": hist,
            "best": best, "final": hist[-1],
            "final_gap_acc": hist[-1]["train_acc"] - hist[-1]["dev_acc"],
            "invasions": inv}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_d, dev_d, cols = load_rte(tok)

    groups = sys.argv[1:] or ["baseline", "A", "B", "C"]
    results = {}
    for g in groups:
        print(f"\n===== group {g} =====", flush=True)
        results[g] = train_one(g, train_d, dev_d, cols, device)
        with open(f"{OUT_DIR}/results_partial.json", "w") as f:
            json.dump(results, f, indent=2)

    # ---- summary plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"baseline": "#2c3e50", "A": "#c0392b", "B": "#27ae60", "C": "#8e44ad"}
    for g, r in results.items():
        axes[0].plot([h["epoch"] for h in r["history"]], [h["dev_acc"] for h in r["history"]],
                     "o-", color=colors[g], label=g)
        axes[1].plot([h["epoch"] for h in r["history"]], [h["train_loss"] for h in r["history"]],
                     "o-", color=colors[g], label=g)
    axes[0].set_title("dev accuracy"); axes[0].set_ylim(0.4, 1.0)
    axes[1].set_title("train loss")
    for a in axes:
        a.set_xlabel("epoch"); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/curves.png", dpi=150)

    # invasion-dimension histogram for group B (the hypothesis group)
    if "B" in results:
        dec = np.zeros(10)
        for inv in results["B"]["invasions"].values():
            dec += np.array(inv["energy_by_decile_of_original_spectrum"])
        dec /= dec.sum()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(range(1, 11), dec, color="#27ae60")
        ax.set_xticks(range(1, 11), [f"{i/10:.1f}" for i in range(1, 11)])
        ax.set_xlabel("decile of original singular-value rank")
        ax.set_ylabel("delta-energy fraction")
        ax.set_title("Where fine-tuning 'invades' (group B, mlp.c_proj)")
        fig.tight_layout(); fig.savefig(f"{OUT_DIR}/invasion_B.png", dpi=150)

    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nfinal dev accuracy:")
    for g, r in results.items():
        print(f"  {g:9s} dev_acc={r['best']['dev_acc']:.3f} (best ep {r['best']['epoch']}) "
              f"train_acc={r['best']['train_acc']:.3f}")
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
