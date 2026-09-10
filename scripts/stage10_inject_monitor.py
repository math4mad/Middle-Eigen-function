"""Stage 10 (0.5B only): online SVD monitoring + active junk-injection dose-response.

Reuses the stage3/5/9 protocol verbatim (imports scripts/stage3_llm_lora.py) so that the
f=0 / cut=none run here is numerically identical to the stage9 baseline runs.

Per optimizer step block (every MON_EVERY steps) we measure, on the monitored
mlp.down_proj matrices, the LoRA increment ΔW = scaling·B@A (rank ≤ LORA_R, so its SVD is
obtained from an r×in problem after a QR of B — cost is milliseconds):

  rel        ‖ΔW_F / ‖W₀‖_F                     — how much the run is rewriting the layer
  r_eff      exp(entropy of σ²/Σσ²)               — effective rank of the update (new axes?)
  top_E      energy of ΔW row-space in decile 1   — 入侵"刚性骨架"的程度
  tail_E     energy in deciles 6–10               — 入侵"松散谱/尾部噪声"的程度  ← 垃圾注入的主指标
  cos_max    max |cos| of ΔW directions to W₀ directions
  angle      mean arcsin of that alignment (子空间旋转角)

Experiment grid (env-configurable):
  JUNK_FRACS  fraction of the training stream replaced by "不当信息"
              (= RTE *test* pairs, which carry no gold label, given random labels → plausible
              sentences with contradictory supervision)
  SEEDS       training seeds (paired by f) — needed because the effect we care about is ~0.03
  CUTS        'none' or 'TAIL6' (stage9's deep-layer top-10% cut) → does the cut recover more
              damage as junk grows?
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
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s3", f"{HERE}/stage3_llm_lora.py")
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.abspath(
    "models/models/Qwen--Qwen2.5-0.5B/snapshots/master")
RTE_DIR = os.environ.get("RTE_DIR", "data/RTE")
JUNK_TSV = os.environ.get("JUNK_TSV", f"{RTE_DIR}/test.tsv")
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage10")
FRAC_TRAIN = float(os.environ.get("FRAC_TRAIN", "0.10"))
DATA_SEED = int(os.environ.get("DATA_SEED", "42"))
JUNK_SEED = int(os.environ.get("JUNK_SEED", "999"))
JUNK_FRACS = [float(x) for x in os.environ.get("JUNK_FRACS", "0,0.1,0.25,0.5").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2").split(",")]
CUTS = os.environ.get("CUTS", "none").split(",")
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH = int(os.environ.get("BATCH", "16"))
MON_EVERY = int(os.environ.get("MON_EVERY", "4"))
MON_LAYERS = [int(x) for x in os.environ.get("MON_LAYERS", "0,6,12,17,18,20,22,23").split(",")]

s3.MODEL_NAME = MODEL_PATH
s3.RTE_DIR = RTE_DIR
s3.FRAC_TRAIN = FRAC_TRAIN
s3.DATA_SEED = DATA_SEED
s3.EPOCHS = EPOCHS
s3.BATCH = BATCH


# ---------------- junk stream ----------------
def make_junk(tokenizer, n, rng):
    import pandas as pd
    df = pd.read_csv(JUNK_TSV, sep="\t")
    take = rng.choice(len(df), size=min(n, len(df)), replace=False)
    sub = df.iloc[take]
    labels = rng.integers(0, 2, size=len(sub))
    enc = tokenizer(sub["sentence1"].tolist(), sub["sentence2"].tolist(),
                    truncation=True, max_length=128, padding="max_length")
    cols = [c for c in ("input_ids", "attention_mask", "token_type_ids") if c in enc]
    tens = [torch.tensor(enc[c]) for c in cols] + [torch.tensor(labels).long()]
    return TensorDataset(*tens), cols


def mix(clean, junk, cols):
    if junk is None:
        return clean
    a = [torch.cat([clean.tensors[i], junk.tensors[i]]) for i in range(len(clean.tensors))]
    return TensorDataset(*a)


# ---------------- online spectral monitor ----------------
class Monitor:
    """Holds the pre-ablation (i.e. run-start) spectrum of the monitored layers.
    Module refs must be re-bound after inject_lora() replaces nn.Linear with loralib.Linear."""
    def __init__(self, model, layers):
        self.names = [f"model.layers.{i}.mlp.down_proj" for i in layers]
        self.mods, self.base = [], {}
        named = dict(model.named_modules())
        for name in self.names:
            mod = named.get(name)
            if mod is None:
                continue
            W = mod.weight.detach().float().cpu().numpy()
            _, s0, V0t = np.linalg.svd(W, full_matrices=False)
            self.base[name] = {"W0": W, "s0": s0, "V0t": V0t, "fro": float(np.linalg.norm(W)),
                               "n": len(s0)}
            self.mods.append((name, mod))

    def rebind(self, model):
        named = dict(model.named_modules())
        self.mods = [(n, named[n]) for n, _ in self.mods if n in named]

    @torch.no_grad()
    def sample(self, step, epoch, loss):
        out = {"step": step, "epoch": epoch, "loss": float(loss)}
        acc = {}
        for name, mod in self.mods:
            B = mod.lora_B.detach().float().cpu()          # [out, r]
            A = mod.lora_A.detach().float().cpu()          # [r, in]
            Q, R = torch.linalg.qr(B)                      # B = Q R, Q orthonormal
            M = (R @ A) * mod.scaling                      # [r, in] same singular values as B@A
            _, s, Vh = torch.linalg.svd(M, full_matrices=False)
            s = s.numpy().astype(np.float64)
            Vh = Vh.numpy().astype(np.float64)
            b = self.base[name]
            p = (s ** 2) / max((s ** 2).sum(), 1e-20)
            r_eff = float(np.exp(-(p * np.log(np.maximum(p, 1e-20))).sum()))
            cos = np.abs(Vh @ b["V0t"].T)                  # [r, n]
            dec = np.minimum((cos.argmax(1) / b["n"] * 10).astype(int), 9)
            hist = np.zeros(10)
            np.add.at(hist, dec, p)
            rel = float(np.sqrt((s ** 2).sum()) / max(b["fro"], 1e-12))
            amax = np.arcsin(np.clip(cos.max(1)[:8], 0, 1))
            for k, v in dict(rel=rel, r_eff=r_eff, top_E=float(hist[0]),
                             mid_E=float(hist[1:5].sum()), tail_E=float(hist[5:].sum()),
                             cos_max=float(cos.max(1)[0]), angle=float(amax.mean())).items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            out[k] = float(np.mean(v))
        out["tail_E_max_layer"] = float(max(acc["tail_E"]))
        return out


# ---------------- one run ----------------
def train_one(frac_junk, seed, cut, tok, clean_d, dev_d, cols, device, mon_path):
    s3.SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng_j = np.random.default_rng(JUNK_SEED)
    n_clean = len(clean_d)
    n_junk = int(round(n_clean * frac_junk / (1 - frac_junk))) if frac_junk > 0 else 0
    junk_d, _ = make_junk(tok, n_junk, rng_j) if n_junk else (None, None)
    train_d = mix(clean_d, junk_d, cols)
    n_mix = len(train_d)

    model = s3.AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=2, dtype=torch.float32)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id or 0
    model.config.problem_type = "single_label_classification"
    ab = s3.ablate_model(model, "baseline" if cut in ("none", "baseline") else cut)
    mon = Monitor(model, MON_LAYERS)
    model = s3.inject_lora(model)
    mon.rebind(model)                      # loralib wrappers replaced the tracked modules
    trainable = [p for p in model.parameters() if p.requires_grad]
    model = model.to(device)
    tl = DataLoader(train_d, batch_size=BATCH, shuffle=True)
    dl_ = DataLoader(dev_d, batch_size=BATCH)
    optim = torch.optim.AdamW(trainable, lr=s3.LR)

    hist, samples, step = [], [], 0
    with open(mon_path, "w") as mf:
        mf.write(json.dumps({"meta": {"junk_frac": frac_junk, "seed": seed, "cut": cut,
                                      "n_clean": n_clean, "n_junk": n_junk, "n_mix": n_mix,
                                      "layers": MON_LAYERS, "mon_every": MON_EVERY}}) + "\n")
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            run_loss, run_n = 0.0, 0
            for b in tl:
                b = {c: t.to(device) for c, t in zip(cols + ["labels"], b)}
                y = b.pop("labels")
                loss = nn.functional.cross_entropy(model(**b).logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step(); optim.zero_grad()
                step += 1
                run_loss += loss.item(); run_n += 1
                if step % MON_EVERY == 0:
                    rec = mon.sample(step, ep + 1, run_loss / run_n)
                    samples.append(rec)
                    mf.write(json.dumps(rec) + "\n"); mf.flush()
            trl, tra = s3.evaluate(model, tl, device, cols)
            dvl, dva = s3.evaluate(model, dl_, device, cols)
            hist.append({"epoch": ep + 1, "train_loss": trl, "train_acc": tra,
                         "dev_loss": dvl, "dev_acc": dva})
            last = samples[-1] if samples else {}
            print(f"[f={frac_junk:.2f} seed={seed} cut={cut}] ep{ep+1} "
                  f"train_acc={tra:.3f} dev_acc={dva:.3f} dev_loss={dvl:.2f} "
                  f"tail_E={last.get('tail_E', float('nan')):.3f} "
                  f"r_eff={last.get('r_eff', float('nan')):.2f} ({time.time()-t0:.0f}s)", flush=True)
    inv = {}
    for n, m in model.named_modules():
        if n in mon.base and isinstance(m, loralib.Linear):
            inv[n] = s3.invasions(mon.base[n]["W0"], s3.lora_delta(m))
    best = max(hist, key=lambda h: h["dev_acc"])
    return {"junk_frac": frac_junk, "seed": seed, "cut": cut, "n_clean": n_clean,
            "n_junk": n_junk, "history": hist, "best": best, "final": hist[-1],
            "monitor": samples, "invasions": inv, "runtime_s": time.time() - t0}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, "model:", MODEL_PATH, flush=True)
    tok = s3.AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    clean_d, dev_d, cols = s3.load_rte(tok)
    runs = {}
    for cut in CUTS:
        for seed in SEEDS:
            for f in JUNK_FRACS:
                tag = f"f{int(round(f*100)):03d}_{cut}_s{seed}"
                p = f"{OUT_DIR}/results_partial.json"
                if os.path.exists(p):
                    runs = json.load(open(p))
                    if tag in runs:
                        continue
                print(f"\n===== run {tag} =====", flush=True)
                runs[tag] = train_one(f, seed, cut, tok, clean_d, dev_d, cols, device,
                                      f"{OUT_DIR}/monitor_{tag}.jsonl")
                json.dump(runs, open(p, "w"), indent=2)
    json.dump(runs, open(f"{OUT_DIR}/results.json", "w"), indent=2)
    print("\nfinal dev accuracy:")
    for tag, r in sorted(runs.items()):
        print(f"  {tag:22s} dev={r['best']['dev_acc']:.3f} junk={r['n_junk']}")


if __name__ == "__main__":
    sys.exit(main())
