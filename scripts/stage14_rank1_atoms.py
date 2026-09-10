"""Stage 14 (Qwen2.5-0.5B): re-parameterize ATTENTION at the rank-1 atom level.

Idea (stage/stage14.md): instead of touching dense input/output halves (LoRA), let the
fine-tuning live on the model's own rank-1 atoms

    W(δ) = W₀ + U_m diag(δ) V_mᵀ ,   δ ∈ R^m trainable,  U_m / V_m FROZEN

* `ATOMm` : U_m, V_m = the top-m SVD atoms of W₀  ⇒ δ literally re-scales existing atoms
            (the "data-science coordinate": σ ordering = functional ordering?)
* `RANDm` : U_m, V_m = a Haar-random orthonormal frame of the same size ⇒ **exactly the same
            capacity (m scalars per matrix), different basis** ⇒ the matched control.
* `ATOMm_L8`: atoms + ordinary LoRA(r=8) on the same matrices ⇒ do we still need *new* directions?
* `lora`  : control = ordinary LoRA r=8 on every projection (the stage3/5/9/10/11 protocol).

Only the attention projections (q,k,v,o) are touched by default (TARGETS=attn); everything else
stays frozen — so `ATOM64` fine-tunes the whole model with 96×64 = 6,144 scalars
(≈700× fewer trainable parameters than LoRA r=8).
"""
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, f"{HERE}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s3 = _load("stage3_llm_lora")

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.abspath(
    "models/models/Qwen--Qwen2.5-0.5B/snapshots/master")
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage14")
RTE_DIR = os.environ.get("RTE_DIR", "data/RTE")
FRAC_TRAIN = float(os.environ.get("FRAC_TRAIN", "0.10"))
DATA_SEED = int(os.environ.get("DATA_SEED", "42"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH = int(os.environ.get("BATCH", "16"))
TARGETS = os.environ.get("TARGETS", "attn")          # attn | mlp | both
LORA_R = int(os.environ.get("LORA_R", "8"))

s3.MODEL_NAME = MODEL_PATH
s3.RTE_DIR = RTE_DIR
s3.FRAC_TRAIN = FRAC_TRAIN
s3.DATA_SEED = DATA_SEED
s3.EPOCHS = EPOCHS
s3.BATCH = BATCH

PAT = {"attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
       "mlp": ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"),
       "both": ("q_proj", "k_proj", "v_proj", "o_proj", "mlp.down_proj")}[TARGETS]
GRP_RE = re.compile(r"^ATOM(\d+|full)(?:_L(\d+))?$")
RND_RE = re.compile(r"^RAND(\d+|full)(?:_L(\d+))?$")


class AtomLinear(nn.Module):
    """frozen W₀  +  U diag(δ ⊙ scale) Vᵀ  (δ trainable, dimensionless)  [+ optional LoRA pair]

    `scale` makes δ comparable across the two bases:
      ATOM: scale = the selected σ  ⇒ δ_i = "把第 i 个原子改变 δ_i 倍"（相对重标）
      RAND: scale = mean(σ_m)       ⇒ 同一数量级的各向同性尺度
    """

    def __init__(self, lin, U, V, scale, lora_r=0):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.register_buffer("weight", lin.weight.detach().clone())
        self.register_buffer("bias", lin.bias.detach().clone() if lin.bias is not None else torch.zeros(0))
        self.register_buffer("U", torch.from_numpy(U).float())          # [out, m]
        self.register_buffer("V", torch.from_numpy(V).float())          # [in, m]
        self.register_buffer("scale", torch.from_numpy(scale).float())  # [m]
        self.delta = nn.Parameter(torch.zeros(U.shape[1]))
        self.lora_r = lora_r
        if lora_r:
            A = torch.empty(lora_r, self.in_features)
            nn.init.kaiming_uniform_(A, a=5 ** 0.5)
            self.lora_A = nn.Parameter(A)
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, lora_r))
            self.scaling = 1.0 / lora_r
        self.merged = False

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias if self.bias.numel() else None)
        y = y + ((x @ self.V) * (self.delta * self.scale)) @ self.U.T
        if self.lora_r:
            y = y + ((x @ self.lora_A.T) * self.scaling) @ self.lora_B.T
        return y


def atom_frames(W, m, random_basis, seed):
    """Return (U_m [out,m], V_m [in,m], scale [m]) — SVD atoms or a Haar-random orthonormal frame."""
    out, inn = W.shape
    m = min(m, min(out, inn))
    s = np.linalg.svd(W, full_matrices=False)[1]
    if random_basis:
        rng = np.random.default_rng(4000 + seed)
        Vm, _ = np.linalg.qr(rng.standard_normal((inn, m)))        # [in, m] orthonormal
        Um, _ = np.linalg.qr(rng.standard_normal((out, m)))        # [out, m] orthonormal
        return Um, Vm, np.full(m, s[:m].mean())
    U, sv, Vt = np.linalg.svd(W, full_matrices=False)
    return U[:, :m], Vt[:m].T, sv[:m]


def build(group, model, seed):
    """Replace target matrices with AtomLinear; returns (#replaced, m, extra LoRA r)."""
    if group == "lora":
        s3.inject_lora(model)
        return 0, None, None
    m_r, extra, rnd = None, 0, False
    for rex, is_rand in ((GRP_RE, False), (RND_RE, True)):
        mt = rex.match(group)
        if mt:
            m_r = model.config.hidden_size if mt.group(1) == "full" else int(mt.group(1))
            extra = int(mt.group(2) or 0)
            rnd = is_rand
            break
    assert m_r, f"unknown group {group}"
    reps = [(n, mod) for n, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and any(p in n for p in PAT) and not n.startswith("score")]
    for n, mod in reps:
        U, V, sc = atom_frames(mod.weight.detach().float().numpy(), m_r, rnd, seed)
        new = AtomLinear(mod, U, V, sc, lora_r=extra)
        parent, attr = model, n.split(".")
        for a in attr[:-1]:
            parent = getattr(parent, a)
        setattr(parent, attr[-1], new)
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if n.startswith("score") or ".delta" in n or "lora_B" in n or "lora_A" in n:
            p.requires_grad_(True)
    return len(reps), m_r, extra


def train_one(group, seed, train_d, dev_d, cols, device):
    s3.SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = s3.AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=2, dtype=torch.float32)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id or 0
    model.config.problem_type = "single_label_classification"
    n_rep, m_r, extra = build(group, model, seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_par = int(sum(p.numel() for p in trainable))
    print(f"group {group}: targets={TARGETS} replaced={n_rep} m={m_r} extraLoRA={extra} "
          f"trainable={n_par:,}", flush=True)
    model = model.to(device)
    tl = DataLoader(train_d, batch_size=BATCH, shuffle=True)
    dl_ = DataLoader(dev_d, batch_size=BATCH)
    # δ 是无量纲的"相对重标"⇒ 原子系数需要比 LoRA 大得多的 lr；分类头仍用原协议 lr
    lr_atom = float(os.environ.get("LR", "0")) or (2e-4 if group == "lora" else 1e-2)
    lr_head = float(os.environ.get("LR_HEAD", "2e-4"))
    atom_p = [p for n, p in model.named_parameters() if p.requires_grad and ".delta" in n]
    rest_p = [p for p in trainable if p not in set(atom_p)]
    print(f"lr_atom={lr_atom:g} lr_rest={lr_head:g} (group={group}, "
          f"{sum(p.numel() for p in atom_p):,} atom scalars / {sum(p.numel() for p in rest_p):,} other)",
          flush=True)
    optim = torch.optim.AdamW(
        [{"params": atom_p, "lr": lr_atom}, {"params": rest_p, "lr": lr_head}] if atom_p else
        [{"params": rest_p, "lr": lr_atom}], lr=lr_atom)
    hist, t0 = [], time.time()
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
        print(f"[{group} s{seed}] ep{ep+1} train_acc={tra:.3f} dev_acc={dva:.3f} "
              f"dev_loss={dvl:.2f} ({time.time()-t0:.0f}s)", flush=True)
    best = max(hist, key=lambda h: h["dev_acc"])
    return {"group": group, "seed": seed, "targets": TARGETS, "m": m_r, "extra_lora": extra,
            "lr_atom": lr_atom, "lr_rest": lr_head,
            "n_replaced": n_rep, "trainable": n_par, "history": hist, "best": best, "final": hist[-1]}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, "model:", MODEL_PATH, "targets:", TARGETS, flush=True)
    tok = s3.AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_d, dev_d, cols = s3.load_rte(tok)
    groups = sys.argv[1:] or ["ATOM64", "RAND64", "ATOMfull", "lora"]
    p = f"{OUT_DIR}/results_partial.json"
    results = json.load(open(p)) if os.path.exists(p) else {}
    for seed in SEEDS:
        for g in groups:
            tag = f"{g}_s{seed}"
            if tag in results:
                continue
            print(f"\n===== run {tag} =====", flush=True)
            results[tag] = train_one(g, seed, train_d, dev_d, cols, device)
            json.dump(results, open(p, "w"), indent=2)
    json.dump(results, open(f"{OUT_DIR}/results.json", "w"), indent=2)
    print("\nfinal dev accuracy:")
    for tag, r in sorted(results.items()):
        print(f"  {tag:18s} dev={r['best']['dev_acc']:.3f} trainable={r['trainable']:,}")


if __name__ == "__main__":
    sys.exit(main())
