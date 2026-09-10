"""Stage 11 (Qwen2.5-0.5B): does the *orientation* of the learning subspace matter,
or only its *dimension*?  Random-subspace fine-tuning vs SVD-index slicing.

Idea (stage/stage11.md): slicing the spectrum by singular-value rank conflates "energy"
with "functional role".  The rotation-invariant alternative is to draw a Haar-random
orthonormal subspace and confine the parameter update to it:

    ΔW = s · B @ A_rand ,   A_rand ∈ R^{d×in} with orthonormal rows (FROZEN, from QR of a
                              random Gaussian),  B ∈ R^{out×d} trainable

⇒ the update's row space is a *random* d-dim slice of the input space, entirely independent
of the SVD ordering, while the output directions are still learned.

Groups (argv):
    lora              standard LoRA (both A and B trainable) — control, same protocol as stage3/5/9/10
    RS<d>             random-subspace LoRA with dimension d  (d=8 → trainable params = 0.85× of LoRA r=8;
                      d=10 → param-matched: 48,640 vs 46,080)
    RS<d>:<scope>     only apply the random subspace on layers in scope (TAILn / FRONTn / MIDn),
                      the remaining layers keep ordinary trainable LoRA
Every run also records the stage-10 online spectral monitor (rel / r_eff / top_E / mid_E /
tail_E / angle) — the prediction for a "spectral fingerprint" of random-subspace training is
tail_E ↑ (energy spread over the whole original spectrum instead of the top decile).
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
OUT_DIR = os.environ.get("OUT_DIR", "outputs/stage11")
RTE_DIR = os.environ.get("RTE_DIR", "data/RTE")
FRAC_TRAIN = float(os.environ.get("FRAC_TRAIN", "0.10"))
DATA_SEED = int(os.environ.get("DATA_SEED", "42"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH = int(os.environ.get("BATCH", "16"))
LORA_R = int(os.environ.get("LORA_R", "8"))
MON_EVERY = int(os.environ.get("MON_EVERY", "4"))
MON_LAYERS = [int(x) for x in os.environ.get("MON_LAYERS", "0,6,12,17,18,20,22,23").split(",")]

for m in (s3, s10):
    m.MODEL_NAME = MODEL_PATH
    m.MODEL_PATH = MODEL_PATH
    m.RTE_DIR = RTE_DIR
    m.FRAC_TRAIN = FRAC_TRAIN
    m.DATA_SEED = DATA_SEED
    m.EPOCHS = EPOCHS
    m.BATCH = BATCH
    m.LORA_R = LORA_R
s3.AB_SUBSTR = "mlp.down_proj"

RS_RE = re.compile(r"^RS(\d+)(?::(TAIL\d+|FRONT\d+|MID\d+))?$")
SHUF_RE = re.compile(r"^SHUF(b|g|inv)(?::(TAIL\d+|FRONT\d+|MID\d+))?$")


def shuffle_spectrum(model, mode, scope, seed):
    """Experiment B: permute the σ↔direction pairing of mlp.down_proj while keeping the
    singular-value multiset (hence the whole spectrum, energy profile, Frobenius norm and rank)
    EXACTLY intact.  Any downstream change is then attributable to the *ordering* itself.

      b   = permute σ inside each decile band
      g   = permute σ globally   (if ordering carries nothing, performance survives)
      inv = reverse σ            (extreme control: strongest energy on weakest directions)
    """
    rng = np.random.default_rng(7000 + seed)
    diag = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not (name.endswith("mlp.down_proj") and isinstance(mod, nn.Linear)):
                continue
            li = int(name.split("layers.")[1].split(".")[0])
            if scope is not None and li not in scope:
                continue
            W = mod.weight.detach().float().numpy().copy()   # copy(): .numpy() aliases the tensor
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            sp = s.copy()
            if mode == "g":
                rng.shuffle(sp)
            elif mode == "b":
                edges = np.linspace(0, len(s), 11).astype(int)
                for a, b in zip(edges[:-1], edges[1:]):
                    blk = s[a:b].copy()
                    rng.shuffle(blk)
                    sp[a:b] = blk
            elif mode == "inv":
                sp = s[::-1].copy()
            else:
                raise ValueError(mode)
            Wn = (U * sp) @ Vt
            mod.weight.copy_(torch.from_numpy(Wn).to(mod.weight.dtype))
            diag[name] = {"rel_change": float(np.linalg.norm(W - Wn) / np.linalg.norm(W)),
                          "sigma_sum_err": float(abs(sp.sum() - s.sum()) / s.sum())}
    return diag


def parse_group(name, n_layers):
    """→ (kind, arg, scope)  with kind ∈ {'lora','rs','shuf'}"""
    if name == "lora":
        return "lora", None, None
    m = RS_RE.match(name)
    if m:
        scope = None if m.group(2) is None else set(s3.dyn_layer_set(m.group(2), n_layers))
        return "rs", int(m.group(1)), scope
    m = SHUF_RE.match(name)
    if m:
        scope = None if m.group(2) is None else set(s3.dyn_layer_set(m.group(2), n_layers))
        return "shuf", m.group(1), scope
    raise ValueError(name)


def inject_lora_rs(model, d, scope):
    """Trainable LoRA everywhere; where in scope, freeze A to a Haar-random orthonormal
    [d × in] matrix (QR of a Gaussian).  Returns #trainable params."""
    rng = torch.Generator().manual_seed(RS_SEED.value)
    reps = [(n, mod) for n, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and not n.startswith("score")]
    frozen_A = []
    for name, mod in reps:
        li = int(name.split("layers.")[1].split(".")[0]) if "layers." in name else -1
        use_rs = d is not None and (scope is None or li in scope)
        r = d if use_rs else LORA_R
        new = loralib.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None, r=r)
        new.weight.data.copy_(mod.weight.data)
        if mod.bias is not None:
            new.bias.data.copy_(mod.bias.data)
        if use_rs:
            G = torch.randn(mod.in_features, r, generator=rng)      # in >= r required
            Q, _ = torch.linalg.qr(G)                               # Q: [in, r] orthonormal cols
            frozen_A.append(f"{name}.lora_A")
            with torch.no_grad():
                new.lora_A.copy_(Q.T)                               # A: [r, in]
            new.lora_A.requires_grad_(False)
        parent, attr = model, name.split(".")
        for a in attr[:-1]:
            parent = getattr(parent, a)
        setattr(parent, attr[-1], new)
    loralib.mark_only_lora_as_trainable(model, bias="none")
    for n, p in model.named_parameters():
        if n.startswith("score"):
            p.requires_grad_(True)
        elif n in set(frozen_A):                                    # re-freeze the random A's
            p.requires_grad_(False)
    return len(frozen_A), len(reps)


class _SeedBox:
    value = 1234


RS_SEED = _SeedBox()


def train_one(group, seed, train_d, dev_d, cols, device, mon_path):
    global RS_SEED
    s3.SEED = seed
    RS_SEED.value = 1000 + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = s3.AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=2, dtype=torch.float32)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id or 0
    model.config.problem_type = "single_label_classification"
    ab = s3.ablate_model(model, "baseline")                  # no spectral slicing this stage
    n_layers = len({n.split("layers.")[1].split(".")[0] for n, _ in model.named_modules()
                    if "mlp.down_proj" in n})
    kind, arg, scope = parse_group(group, n_layers)
    shuf_diag = {}
    if kind == "shuf":
        shuf_diag = shuffle_spectrum(model, arg, scope, seed)
        vals = [v["rel_change"] for v in shuf_diag.values()]
        print(f"group {group}: SHUF mode={arg} scope={'all' if scope is None else sorted(scope)} "
              f"mean‖W'-W‖/‖W‖={np.mean(vals):.3f} over {len(vals)} layers "
              f"(σ 多重集保持不变, 最大误差 {max(v['sigma_sum_err'] for v in shuf_diag.values()):.1e})", flush=True)
        d, scope_rs = None, None
    else:
        d, scope_rs = (arg if kind == "rs" else None), scope
    mon = s10.Monitor(model, MON_LAYERS)
    n_rs, n_all = inject_lora_rs(model, d, scope_rs)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"group {group}: kind={kind} d={d} scope={'all' if scope is None else sorted(scope)} "
          f"RS-proj={n_rs}/{n_all} (层范围={n_layers} 层) trainable={sum(p.numel() for p in trainable):,}", flush=True)
    model = model.to(device)
    mon.rebind(model)
    tl = DataLoader(train_d, batch_size=BATCH, shuffle=True)
    dl_ = DataLoader(dev_d, batch_size=BATCH)
    optim = torch.optim.AdamW(trainable, lr=s3.LR)

    hist, samples, step = [], [], 0
    t0 = time.time()
    with open(mon_path, "w") as mf:
        mf.write(json.dumps({"meta": {"group": group, "seed": seed, "kind": kind, "arg": arg,
                                      "scope": None if scope is None else sorted(scope),
                                      "shuf_diag": shuf_diag,
                                      "n_rs_proj": n_rs, "n_layers": n_layers,
                                      "trainable": int(sum(p.numel() for p in trainable))}}) + "\n")
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
            print(f"[{group} s{seed}] ep{ep+1} train_acc={tra:.3f} dev_acc={dva:.3f} "
                  f"dev_loss={dvl:.2f} tail_E={last.get('tail_E', float('nan')):.3f} "
                  f"top_E={last.get('top_E', float('nan')):.3f} ({time.time()-t0:.0f}s)", flush=True)
    inv = {}
    for n, m_ in model.named_modules():
        if n in mon.base and isinstance(m_, loralib.Linear):
            inv[n] = s3.invasions(mon.base[n]["W0"], s3.lora_delta(m_))
    best = max(hist, key=lambda h: h["dev_acc"])
    return {"group": group, "seed": seed, "kind": kind, "arg": arg,
            "scope": None if scope is None else sorted(scope), "n_rs_proj": n_rs,
            "shuf_rel_change": float(np.mean([v["rel_change"] for v in shuf_diag.values()])) if shuf_diag else 0.0,
            "trainable": int(sum(p.numel() for p in trainable)), "history": hist,
            "best": best, "final": hist[-1], "invasions": inv, "runtime_s": time.time() - t0}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, "model:", MODEL_PATH, flush=True)
    tok = s3.AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_d, dev_d, cols = s3.load_rte(tok)
    groups = sys.argv[1:] or ["lora", "RS8", "RS10", "RS8:TAIL6", "RS8:M12",
                              "SHUFb", "SHUFg", "SHUFinv"]
    results = {}
    p = f"{OUT_DIR}/results_partial.json"
    if os.path.exists(p):
        results = json.load(open(p))
    for seed in SEEDS:
        for g in groups:
            tag = f"{g}_s{seed}"
            if tag in results:
                continue
            print(f"\n===== run {tag} =====", flush=True)
            results[tag] = train_one(g, seed, train_d, dev_d, cols, device,
                                    f"{OUT_DIR}/monitor_{tag}.jsonl")
            json.dump(results, open(p, "w"), indent=2)
    json.dump(results, open(f"{OUT_DIR}/results.json", "w"), indent=2)
    print("\nfinal dev accuracy:")
    for tag, r in sorted(results.items()):
        print(f"  {tag:16s} dev={r['best']['dev_acc']:.3f} trainable={r['trainable']:,}")


if __name__ == "__main__":
    sys.exit(main())
