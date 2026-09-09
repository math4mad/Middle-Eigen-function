"""Stage 6: 小模型 + "junk DNA" 惰性参数 → 大模型能力？

构造：对 encoder 每一层 L_i，插入一个权重复制自 L_i 的"惰性副本" J_i，
前向为  h <- L_i(h) ;  h <- h + tanh(a_i) * J_i(h)，a_i 初始化为 0。
  - 初始化时输出与原模型完全一致（函数严格保持），但参数量 ~2x
  - 惰性块 = 生物"非编码 DNA"：在场、初始无功能、训练中可被征用（a_i 长大）
组：
  expand_recruit : a_i 与 J_i 全部可训练（征用 junk）
  expand_frozen  : a_i 冻结为 0（junk 存在但永远不可达）——应等价于基线（管道对照）
  baseline       : 不扩展
评估与 stage3 完全一致（同 249 行 RTE 子集、lr、epochs）。
训练后额外记录：每层 |tanh(a_i)|（征用强度）、junk 块 ΔW 的谱位置。
"""
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3_ablate_train as S3  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.abspath("models/AI-ModelScope/bert-base-uncased")
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage6_expand")
SEED = 42


GATE_LR = float(os.environ.get("GATE_LR", 0.0))   # if >0: separate high-lr group for gates


class GatedTwin(nn.Module):
    """real layer + junk twin gated by tanh(alpha), alpha starts at 0."""

    def __init__(self, real, junk, trainable_gate=True):
        super().__init__()
        self.real = real
        self.junk = junk
        self.trainable_gate = trainable_gate
        if trainable_gate:
            # MUST be nn.Parameter so model.parameters()/optimizer pick it up
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, *args, **kwargs):
        h = self.real(hidden_states, *args, **kwargs)
        if not self.trainable_gate:  # frozen gate == exact identity (junk never called)
            return h
        j = self.junk(h, *args, **kwargs)
        return h + torch.tanh(self.alpha) * j


def expand_model(model, trainable_gate):
    enc = model.bert.encoder
    originals = list(enc.layer)          # snapshot: never mutate-while-iterating
    twins = [GatedTwin(orig, copy.deepcopy(orig), trainable_gate=trainable_gate)
             for orig in originals]
    enc.layer = nn.ModuleList(twins)
    return model


def gate_growth(model):
    return {f"layer.{i}": float(torch.tanh(l.alpha).abs())
            for i, l in enumerate(model.bert.encoder.layer) if isinstance(l, GatedTwin)}


def junk_delta_spectra(model, w0):
    """For each junk block matrix, decile position of training ΔW energy in
    the junk block's own initial spectrum (same 'invasion' machinery)."""
    out = {}
    for name, p in model.named_parameters():
        if ".junk." not in name or not name.endswith("weight") or p.ndim != 2:
            continue
        base = name.replace(".junk.", ".")          # real-layer param key
        w1 = p.detach().float().cpu().numpy()
        out[name] = S3.invasions(w0[name], w1)
    return out


def train_group(group, device):
    S3.set_all_seeds()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    if group != "baseline":
        model = expand_model(model, trainable_gate=group.startswith("expand_recruit"))
    # snapshot of junk weights for later ΔW spectra
    w0 = {n: p.detach().float().cpu().numpy().copy()
          for n, p in model.named_parameters() if ".junk." in n and n.endswith("weight") and p.ndim == 2}

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    train_d, dev_d, cols = S3.load_rte(tok)
    tl = DataLoader(train_d, batch_size=S3.BATCH, shuffle=True)
    dl_ = DataLoader(dev_d, batch_size=S3.BATCH)
    model = model.to(device)
    gates = [p for n, p in model.named_parameters() if n.endswith(".alpha")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".alpha")]
    pg = [{"params": rest, "lr": S3.LR}]
    if gates and GATE_LR > 0:
        pg.append({"params": gates, "lr": GATE_LR})
    optim = torch.optim.AdamW(pg, weight_decay=0.01)
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
    best = max(hist, key=lambda h: h["dev_acc"])
    return {"group": group, "history": hist, "best": best, "final": hist[-1],
            "gates": gate_growth(model),
            "junk_invasions": junk_delta_spectra(model, w0) if w0 else {},
            "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    # ---- output-equivalence check: wrap the SAME instance, classifier head not re-init ----
    if os.environ.get("SKIP_EQUIV", "0") != "1":
        torch.manual_seed(SEED)
        base = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(device).eval()
        x = torch.randint(3, 29000, (2, 128), device=device)
        at = torch.ones_like(x, device=device)
        tt = torch.zeros_like(x, device=device)
        with torch.no_grad():
            l0 = base(input_ids=x, attention_mask=at, token_type_ids=tt).logits.clone()
            exp = expand_model(base, trainable_gate=True).to(device).eval()
            l1 = exp(input_ids=x, attention_mask=at, token_type_ids=tt).logits
            d = (l0 - l1).abs().max().item()
        print(f"init output-equivalence max|Δlogit| = {d:.2e}", flush=True)
        assert d < 1e-3, "expansion must preserve function at init"
        del base, exp
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()

    results = {}
    rf = f"{OUT_DIR}/results.json"
    if os.path.exists(rf):
        results = json.load(open(rf))
    for g in (sys.argv[1:] or ["expand_recruit", "expand_frozen"]):
        if g in results:
            print("skip", g); continue
        print(f"\n===== {g} =====", flush=True)
        results[g] = train_group(g, device)
        with open(rf, "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps({k: round(v["best"]["dev_acc"], 3) for k, v in results.items()}))
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
