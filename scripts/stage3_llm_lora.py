"""Stage 3-LLM: SVD ablation on Qwen2.5-0.5B mlp.down_proj + LoRA fine-tuning on RTE.

Differences from the BERT pipeline:
  - ablation target: every `mlp.down_proj` (the LLM analogue of mlp.c_proj)
  - fine-tuning: LoRA (loralib, r=8) on all attention + MLP projections;
    base weights frozen (score head trained too)
  - invasion analysis is performed on the *merged* LoRA delta  ΔW = B @ A,
    exactly as AGETNTS.md asks ("如果采用LoRA进行微调，额外记录微调后模型
    权重变化的奇异值谱，观察入侵维度主要分布在哪个区间")

Groups: baseline / A / B / C  (same keep-mask rules as stage3_ablate_train.py)
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
import loralib
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_PATH") or os.path.abspath("models/Qwen/Qwen2.5-0.5B")
RTE_DIR = os.environ.get("RTE_DIR", "data/RTE")
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage3_qwen")
AB_SUBSTR = os.environ.get("ABLATE_SUBSTR", "mlp.down_proj")
SEED = 42
BATCH = int(os.environ.get("BATCH", 16))
EPOCHS = int(os.environ.get("EPOCHS", 5))
LR = float(os.environ.get("LR", 2e-4))
LORA_R = int(os.environ.get("LORA_R", 8))
FRAC_TRAIN = 0.10


def is_target(name):
    return AB_SUBSTR in name


def keep_mask_group(n, group, rng):
    k_top = max(1, int(np.ceil(n * 0.10)))
    k_bot = max(1, int(np.ceil(n * 0.10)))
    mask = np.zeros(n, dtype=bool)
    frac = None
    if group in ("T5", "T20", "T30"):           # dose-response: drop top-f, keep rest
        frac = {"T5": 0.05, "T20": 0.20, "T30": 0.30}[group]
        mask[int(n * frac):] = True
    elif group == "A":
        mask[:k_top] = True
    elif group == "A_e":                          # energy-preserved skeleton
        mask[:k_top] = True
    elif group == "B":
        mask[k_top:n - k_bot] = True
    elif group == "B_e":
        mask[k_top:n - k_bot] = True
    elif group == "C":
        idx = rng.permutation(n)[: int(n * 0.80)]
        mask[idx] = True
    elif group == "baseline":
        mask[:] = True
    else:
        raise ValueError(group)
    return mask, (group.endswith("_e"))


LAYER_SETS = {"E6": range(0, 6), "M12": range(6, 18), "L6": range(18, 24)}


def ablate_model(model, group):
    rng = np.random.default_rng(SEED)
    lset = LAYER_SETS.get(group)
    report = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not (name.endswith(AB_SUBSTR) and isinstance(mod, nn.Linear)):
                continue
            li = int(name.split("layers.")[1].split(".")[0])
            W = mod.weight.detach().float().numpy()
            if lset is not None:
                # layer-targeted: drop top-10% ONLY on layers in the set, keep others intact
                if li not in lset:
                    report[name] = {"kept": "intact", "total": W.shape[0], "energy_kept": 1.0}
                    continue
                mask = np.ones(min(W.shape), dtype=bool)
                mask[: max(1, int(np.ceil(min(W.shape) * 0.10)))] = False
                U, s, Vt = np.linalg.svd(W, full_matrices=False)
                S2 = s * mask
            else:
                U, s, Vt = np.linalg.svd(W, full_matrices=False)
                mask, energy_p = keep_mask_group(len(s), group, rng)
                S2 = s * mask
                if energy_p and (S2 ** 2).sum() > 0:
                    S2 = S2 * np.sqrt((s ** 2).sum() / (S2 ** 2).sum())
            Wn = (U * S2) @ Vt
            mod.weight.copy_(torch.from_numpy(Wn).to(mod.weight.dtype))
            report[name] = {"kept": int(mask.sum()), "total": int(mask.size),
                            "energy_kept": float((S2 ** 2).sum() / (s ** 2).sum()),
                            "backfill_gap": float(np.linalg.norm(W - Wn) / np.linalg.norm(W))}
    return report


def inject_lora(model):
    """Wrap every nn.Linear (except score head) in loralib.Linear r=LORA_R."""
    reps = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not name.startswith("score"):
            reps.append((name, mod))
    for name, mod in reps:
        new = loralib.Linear(mod.in_features, mod.out_features,
                             bias=mod.bias is not None, r=LORA_R)
        new.weight.data.copy_(mod.weight.data)
        if mod.bias is not None:
            new.bias.data.copy_(mod.bias.data)
        parent, attr = model, name.split(".")
        for a in attr[:-1]:
            parent = getattr(parent, a)
        setattr(parent, attr[-1], new)
    loralib.mark_only_lora_as_trainable(model, bias="none")
    # the fresh 2-way score head must train too
    for n, p in model.named_parameters():
        if n.startswith("score"):
            p.requires_grad_(True)
    return model


def lora_delta(module):
    """ΔW = scaling * B @ A of a loralib.Linear (the merged weight change)."""
    BA = (module.lora_B @ module.lora_A)  # both are Parameters: [out,r] @ [r,in] = [out,in]
    return BA.detach().float().cpu().numpy() * module.scaling


def invasions(w0, delta):
    _, s0, V0t = np.linalg.svd(w0, full_matrices=False)
    _, _, Vt = np.linalg.svd(delta, full_matrices=False)
    k = min(16, Vt.shape[0])
    cos = np.abs(Vt[:k] @ V0t.T)
    best = cos.max(axis=1)
    rank = cos.argmax(axis=1) / V0t.shape[0]
    wh, _ = np.histogram(rank, bins=10, range=(0, 1), weights=best / best.sum())
    return {"delta_rel_fro": float(np.linalg.norm(delta) / max(np.linalg.norm(w0), 1e-12)),
            "rank_frac_top_delta_dirs": rank[:8].round(3).tolist(),
            "cos_to_original": best[:8].round(3).tolist(),
            "energy_by_decile_of_original_spectrum": wh.round(3).tolist()}


def load_rte(tokenizer):
    import pandas as pd
    lab = {"not_entailment": 0, "entailment": 1}
    def read(fn):
        df = pd.read_csv(f"{RTE_DIR}/{fn}", sep="\t")
        df["label"] = df["label"].map(lab)
        return df.dropna(subset=["label"])
    tr = read("train.tsv").sample(frac=FRAC_TRAIN, random_state=SEED).reset_index(drop=True)
    dv = read("dev.tsv")
    tokenizer.padding_side = "right"
    def feats(df):
        enc = tokenizer(df["sentence1"].tolist(), df["sentence2"].tolist(),
                        truncation=True, max_length=128, padding="max_length")
        cols = [c for c in ("input_ids", "attention_mask", "token_type_ids") if c in enc]
        return TensorDataset(*[torch.tensor(enc[c]) for c in cols],
                             torch.tensor(df["label"].tolist()).long()), cols
    a, cols = feats(tr); b, _ = feats(dv)
    print(f"qwen RTE: train={len(tr)} dev={len(dv)}")
    return a, b, cols


@torch.no_grad()
def evaluate(model, loader, device, cols):
    model.eval()
    tot_l, n, ok = 0.0, 0, 0
    for b in loader:
        b = {c: t.to(device) for c, t in zip(cols + ["labels"], b)}
        y = b.pop("labels")
        logits = model(**b).logits
        tot_l += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        ok += (logits.argmax(-1) == y).sum().item()
        n += y.size(0)
    model.train()
    return tot_l / n, ok / n


def train_one(group, train_d, dev_d, cols, device):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2,
                                                               dtype=torch.float32)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id or 0
    model.config.problem_type = "single_label_classification"  # avoid v5 binary-loss heuristic with num_labels=2
    ab = ablate_model(model, group)
    w0 = {n: m.weight.detach().float().cpu().numpy().copy()
          for n, m in model.named_modules() if n.endswith(AB_SUBSTR) and isinstance(m, nn.Linear)}
    model = inject_lora(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"group {group}: trainable params {sum(p.numel() for p in trainable):,}")
    model = model.to(device)
    tl = DataLoader(train_d, batch_size=BATCH, shuffle=True)
    dl_ = DataLoader(dev_d, batch_size=BATCH)
    optim = torch.optim.AdamW(trainable, lr=LR)
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
        trl, tra = evaluate(model, tl, device, cols)
        dvl, dva = evaluate(model, dl_, device, cols)
        hist.append({"epoch": ep + 1, "train_loss": trl, "train_acc": tra,
                     "dev_loss": dvl, "dev_acc": dva})
        print(f"[{group}] ep{ep+1} train_loss={trl:.3f} train_acc={tra:.3f} "
              f"dev_loss={dvl:.3f} dev_acc={dva:.3f}", flush=True)
    inv = {}
    for n, m in model.named_modules():
        if n.endswith(AB_SUBSTR) and isinstance(m, loralib.Linear):
            inv[n] = invasions(w0[n], lora_delta(m))
    best = max(hist, key=lambda h: h["dev_acc"])
    return {"group": group, "ablation": ab, "history": hist, "best": best, "final": hist[-1],
            "invasions": inv}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, "model:", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_d, dev_d, cols = load_rte(tok)
    groups = sys.argv[1:] or ["baseline", "A", "B", "C"]
    results = {}
    for g in groups:
        print(f"\n===== group {g} =====", flush=True)
        results[g] = train_one(g, train_d, dev_d, cols, device)
        with open(f"{OUT_DIR}/results_partial.json", "w") as f:
            json.dump(results, f, indent=2)
    fig, ax = plt.subplots(figsize=(6, 4))
    for g, r in results.items():
        ax.plot([h["epoch"] for h in r["history"]], [h["dev_acc"] for h in r["history"]], "o-", label=g)
    ax.set_title("Qwen2.5-0.5B + LoRA on 10% RTE"); ax.set_xlabel("epoch")
    ax.set_ylabel("dev acc"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/curves.png", dpi=150)
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nfinal dev accuracy:")
    for g, r in results.items():
        print(f"  {g:9s} dev_acc={r['best']['dev_acc']:.3f} (best ep {r['best']['epoch']})")


if __name__ == "__main__":
    sys.exit(main())
