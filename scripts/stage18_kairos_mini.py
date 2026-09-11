"""Stage 18 · KAIROS-MINI — LoRA injection timing on a from-scratch small LM.

Why from scratch: a pretrained 0.5B has only a FINAL state — T-pre/T-mid have
no k to attach to (the P1 gap, Letters 009/014). This bench trains a small
Qwen2.5-architecture clone (RMSNorm, GQA, RoPE, SwiGLU, tied embeddings)
itself, keeps a real checkpoint ladder k ∈ {0,25,50,75,100}%, and injects
LoRA into a moving base at true epoch points — the LM-scale sibling of
Sarcos's H9-S, at explore scale (~1.6M params, byte-level).

STATUS: EXPLORATORY (schedule (a) only: fixed adapter budget). No prediction
is tested here; outputs are calibration for a future registered H-effect(k)
document. Regimes stay separate rows: k<100 = training-under-constraint
(moving base + adapter on stream B); k=100 = frozen-base LoRA (T-post).
Controls: frozen base (no adaptation) and full-FT on B, both at S steps.

Data: tries TinyStories (hf-mirror) into data/tiny_stories.txt; falls back to
a LOCAL corpus: domain A = this workspace's technical .md files, domain B =
letters/ prose (genuinely different register). Corpus + hashes recorded in
meta.json. Byte-level tokenizer (256) — deterministic, reproducible.

Usage (from the MEF repo root, .venv/bin/python):
  .venv/bin/python scripts/stage18_kairos_mini.py --mode pretrain --steps 600
  .venv/bin/python scripts/stage18_kairos_mini.py --mode sweep --adapter-steps 300
  (smoke: add --steps 20 --adapter-steps 10 --batch 4)
"""
import argparse, json, math, os, time, hashlib, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = int(os.environ.get("SEED", 13))            # canonical Sarcos-family seed
OUT = Path(os.environ.get("OUT_DIR", "outputs/stage18_kairos_mini")); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("TS_PATH", "data/tiny_stories.txt"))
LOCAL_A_GLOBS = ["AGENTS.md", "README.md", "report/**/*.md", "stage/*.md"]
LOCAL_B_GLOBS = ["log/*.log"]                      # different register by construction

# ---------------------------------------------------------------- data
def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

def load_bytes() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (A pretrain bytes, B downstream stream = held-out stories,
    P shift-probe = case-toggled eval, provenance meta).

    B is a TRUE distribution shift for the small model — unseen story text —
    matching the Sarcos shift-task logic (the base should still learn *something*
    there: the floor-effect fence of Letters 009/011, checked, not assumed).
    P tests shift-sensitivity as the forgetting probe complement.
    """
    meta = {}
    if not DATA.exists() or DATA.stat().st_size < 5_000_000:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        try:
            print("fetching TinyStories via hf-mirror …")
            req = urllib.request.Request(
                "https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
                headers={"User-Agent": "stage18/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(DATA, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk: break
                    f.write(chunk)
                    if f.tell() > 40_000_000: break
        except Exception as e:
            print(f"download failed ({e}); local fallback")
    if DATA.exists() and DATA.stat().st_size > 5_000_000:
        raw = DATA.read_bytes()
        n = len(raw); cut, ev = int(n * 0.95), int(n * 0.02)
        A, B = raw[:cut], raw[cut:cut + ev * 2]
        Pb = bytearray(B)
        for i in range(len(Pb)):                    # toggle case: deterministic shift-probe
            if 65 <= Pb[i] <= 90: Pb[i] += 32
            elif 97 <= Pb[i] <= 122: Pb[i] -= 32
        meta = {"kind": "tinystories | B=held-out tail | P=case-toggled-B",
                "shaA": sha(raw[:cut]), "shaB": sha(B), "shaP": sha(bytes(Pb)),
                "bytesA": cut, "bytesB": len(B)}
        return (np.frombuffer(A, dtype=np.uint8), np.frombuffer(B, dtype=np.uint8),
                np.frombuffer(bytes(Pb), dtype=np.uint8), meta)
    # local fallback: technical md corpus (A) vs logs prose mix (B)
    a_txt, b_txt = [], []
    for g in LOCAL_A_GLOBS:
        for p in Path(".").glob(g):
            if p.is_file(): a_txt.append(p.read_text(errors="ignore"))
    for g in LOCAL_B_GLOBS:
        for p in Path(".").glob(g):
            if p.is_file(): b_txt.append(p.read_text(errors="ignore"))
    A = "\n".join(a_txt).encode()[:8_000_000] or b"the quick brown fox " * 40000
    Bsrc = "\n".join(b_txt).encode()[:1_000_000] or A[40000:80000]
    Pb = bytearray(Bsrc)
    for i in range(len(Pb)):
        if 65 <= Pb[i] <= 90: Pb[i] += 32
        elif 97 <= Pb[i] <= 122: Pb[i] -= 32
    meta = {"kind": "local-md | B=logs | P=case-toggled", "shaA": sha(A), "shaB": sha(Bsrc),
            "shaP": sha(bytes(Pb)), "bytesA": len(A), "bytesB": len(Bsrc)}
    return (np.frombuffer(A, dtype=np.uint8), np.frombuffer(Bsrc, dtype=np.uint8),
            np.frombuffer(bytes(Pb), dtype=np.uint8), meta)

def stream(arr: np.ndarray, batch: int, ctx: int, rng):
    while True:
        ix = rng.integers(0, len(arr) - ctx - 1, size=batch)
        x = np.stack([arr[i:i + ctx] for i in ix]).astype(np.int64)
        y = np.stack([arr[i + 1:i + 1 + ctx] for i in ix]).astype(np.int64)
        yield torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)

# ---------------------------------------------------------------- model (Qwen2-style, mini)
class RMSNorm(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d))
    def forward(s, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(v + 1e-6)).to(x.dtype) * s.w

class LoRA(nn.Module):
    """wraps a Linear: y = W x + (x A^T) B^T · scaling. B zero-init (start = identity)."""
    def __init__(s, lin: nn.Linear, r: int, alpha: int = 32):
        super().__init__()
        s.lin, s.r, s.s = lin, r, alpha / r
        d, o = lin.in_features, lin.out_features
        s.A = nn.Parameter(torch.zeros(r, d)); s.B = nn.Parameter(torch.zeros(o, r))
        nn.init.normal_(s.A, std=1.0 / d)
    def forward(s, x):
        return s.lin(x) + ((x @ s.A.T) @ s.B.T) * s.s

def lora_targets(model: nn.Module, r: int):
    targets = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.split(".")[-1] in ("q", "k", "v", "o", "gate", "up", "down")]
    for name in targets:
        parts = name.split("."); parent = model
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRA(getattr(parent, parts[-1]), r))
    model.to(DEV)          # fresh LoRA params init on CPU; return the whole rig to the trainer device
    return targets

class Block(nn.Module):
    def __init__(s, d, nh, nkv, inter):
        super().__init__()
        s.n1, s.n2 = RMSNorm(d), RMSNorm(d)
        s.q, s.k, s.v = nn.Linear(d, nh * 64), nn.Linear(d, nkv * 64), nn.Linear(d, nkv * 64)
        s.o = nn.Linear(nh * 64, d)
        s.gate, s.up, s.down = nn.Linear(d, inter), nn.Linear(d, inter), nn.Linear(inter, d)
        s.nh, s.nkv, s.hd = nh, nkv, 64
    def attn(s, x, mask, cos, sin):
        B, T, _ = x.shape
        q = s.q(x).view(B, T, s.nh, s.hd).transpose(1, 2)
        k = s.k(x).view(B, T, s.nkv, s.hd).transpose(1, 2)
        v = s.v(x).view(B, T, s.nkv, s.hd).transpose(1, 2)
        def rope(t):
            t = t.float(); c, sn = cos[:, :t.shape[-1]], sin[:, :t.shape[-1]]
            x1, x2 = t[..., :t.shape[-1]//2], t[..., t.shape[-1]//2:]
            return (t*c) + (torch.cat((-x2, x1), -1) * sn)
        q, k = rope(q), rope(k)
        k = k.repeat_interleave(s.nh // s.nkv, dim=1); v = v.repeat_interleave(s.nh // s.nkv, dim=1)
        w = (q @ k.transpose(-1, -2)) / math.sqrt(s.hd) + mask
        return s.o((w.softmax(-1).to(q.dtype) @ v).transpose(1, 2).reshape(B, T, -1))
    def forward(s, x, mask, cos, sin):
        x = x + s.attn(s.n1(x), mask, cos, sin)
        h = s.down(F.silu(s.gate(s.n2(x))) * s.up(s.n2(x)))
        return x + h

class MiniQwen(nn.Module):
    def __init__(s, d=192, layers=4, nh=6, nkv=2, inter=512, vocab=256, ctx=256):
        super().__init__()
        s.emb = nn.Embedding(vocab, d)
        nn.init.normal_(s.emb.weight, std=0.02)   # Qwen2.5 initializer_range; std=1 default would put CE ~176
        s.blocks = nn.ModuleList([Block(d, nh, nkv, inter) for _ in range(layers)])
        s.fin = RMSNorm(d); s.ctx = ctx; s.d = d
        inv = 1.0 / (100 ** (torch.arange(0, 64, 2).float() / 64))
        pos = torch.arange(ctx).float()
        ang = torch.outer(pos, inv)
        s.register_buffer("cos", torch.cat([ang.cos(), ang.cos()], -1)[None, None])
        s.register_buffer("sin", torch.cat([ang.sin(), ang.sin()], -1)[None, None])
        s.mask = torch.full((ctx, ctx), float("-inf")).triu(1)
    def forward(s, x):
        h = s.emb(x)
        m = s.mask[:x.shape[1]][None, None].to(h.device)
        c, sn = s.cos[..., :x.shape[1]-1 or 1, :], s.sin[..., :x.shape[1]-1 or 1, :]
        c = F.pad(c, (0, 0, 0, 1)); sn = F.pad(sn, (0, 0, 0, 1))
        for b in s.blocks: h = b(h, m, c, sn)
        return s.fin(h) @ s.emb.weight.T   # tied head

def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---------------------------------------------------------------- trainer
def fit(model, gen, steps, lr, evals: dict, every=50, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    curve = []
    t0 = time.time()
    for i in range(steps):
        x, y = next(gen); opt.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (i + 1) % every == 0 or i == steps - 1:
            ev = {n: evaluate(model, xx, yy) for n, (xx, yy) in evals.items()}
            curve.append({"step": i + 1, "train_loss": float(loss), **ev, "secs": round(time.time() - t0, 1)})
            print(f"{tag} step {i+1}/{steps} loss {float(loss):.3f} " +
                  " ".join(f"{n} {v:.3f}" for n, v in ev.items()))
    return curve

@torch.no_grad()
def evaluate(model, xb, yb):
    model.eval()
    out = float(F.cross_entropy(model(xb).reshape(-1, 256), yb.reshape(-1)))
    model.train()
    return out

def eval_tensors(A, B, P, ctx, n=8, batch=16, rng=None):
    rng = rng or np.random.default_rng(SEED + 1)
    def pack(src):
        ix = rng.integers(0, max(1, len(src) - ctx - 1), size=n * batch)
        x = np.stack([src[i:i+ctx] for i in ix]); y = np.stack([src[i+1:i+1+ctx] for i in ix])
        return torch.from_numpy(x.astype(np.int64)).to(DEV), torch.from_numpy(y.astype(np.int64)).to(DEV)
    return {"A": pack(A), "B": pack(B), "P": pack(P)}

def build(cfg):
    torch.manual_seed(SEED)
    m = MiniQwen(**cfg).to(DEV)
    return m

# ---------------------------------------------------------------- modes
def mode_pretrain(a):
    A, B, P, meta = load_bytes()
    cfg = dict(d=192, layers=4, nh=6, nkv=2, inter=512, ctx=a.ctx)
    model = build(cfg)
    print(f"params {nparams(model):,} · dev {DEV} · corpus {meta['kind']} "
          f"({meta['bytesA']/1e6:.1f}MB A / {meta['bytesB']/1e6:.1f}MB B)")
    rng = np.random.default_rng(SEED)
    ev = eval_tensors(A, B, P, a.ctx)
    ladder = {p: OUT / f"ckpt_k{p}.pt" for p in (0, 25, 50, 75, 100)}
    torch.manual_seed(SEED); model2 = build(cfg)
    rng2 = np.random.default_rng(SEED); gen2 = stream(A, a.batch, a.ctx, rng2)
    opt = torch.optim.AdamW(model2.parameters(), lr=3e-3)
    boundaries = {int(round(p / 100 * a.steps)): p for p in (25, 50, 75)}   # step -> k%
    torch.save(model2.state_dict(), ladder[0])
    curve2 = []
    t0 = time.time()
    for step in range(a.steps):
        x, y = next(gen2); opt.zero_grad()
        loss = F.cross_entropy(model2(x).reshape(-1, 256), y.reshape(-1)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0); opt.step(); i = step + 1
        if i in boundaries:
            torch.save(model2.state_dict(), ladder[boundaries[i]])
            print(f"  latched k={boundaries[i]}% at step {i}")
        if i % max(1, a.steps // 20) == 0:
            curve2.append({"step": i, "loss": float(loss), "secs": round(time.time()-t0, 1)})
    torch.save(model2.state_dict(), ladder[100])
    (OUT / "base_run.json").write_text(json.dumps(
        {"cfg": cfg, "steps": a.steps, "batch": a.batch, "ctx": a.ctx, "seed": SEED,
         "curve": curve2, "meta": meta, "ladder": {str(k): str(v) for k, v in ladder.items()},
         "measured_secs": round(time.time()-t0, 1)}, indent=1))
    (OUT / "eval_A.pt").parent.mkdir(exist_ok=True)
    torch.save(ev["A"], OUT / "eval_A.pt")
    torch.save(ev["B"], OUT / "eval_B.pt")
    torch.save(ev["P"], OUT / "eval_P.pt")
    print("base ladder saved:", ", ".join(map(str, ladder)))

def mode_sweep(a):
    cfg = json.loads((OUT / "base_run.json").read_text())["cfg"]
    A, B, P, _ = load_bytes()
    evA = torch.load(OUT / "eval_A.pt", weights_only=True)
    evB = torch.load(OUT / "eval_B.pt", weights_only=True)
    evP = torch.load(OUT / "eval_P.pt", weights_only=True)
    rng = np.random.default_rng(SEED + 7)
    genB = stream(B, a.batch, cfg["ctx"], rng)          # ALL arms fit on B; k<100 co-trains base (moving), k=100 freezes it
    results = []
    for k in (0, 25, 50, 75, 100):
        for r in ([2, 8, 32] if a.full else [8]):
            model = build(cfg); model.load_state_dict(torch.load(OUT / f"ckpt_k{k}.pt", weights_only=True))
            tagged = lora_targets(model, r)
            if k == 100:
                for n, m in model.named_parameters():
                    if ".A" not in n and ".B" not in n: m.requires_grad = False
            trainable = nparams(model)
            curve = fit(model, genB, a.adapter_steps, 5e-4, {"Bval": evB, "Aval": evA, "Pval": evP},
                        every=max(1, a.adapter_steps // 6), tag=f"k{k} r{r}")
            # ΔW effective rank (merged adapter spectra per tagged module)
            ers = []
            for name, m in model.named_modules():
                if isinstance(m, LoRA):
                    dW = (m.B @ m.A).float() * m.s
                    svs = torch.linalg.svdvals(dW); ers.append(float((svs.sum()**2) / (svs**2).sum()))
            results.append({"k": k, "r": r, "trainable": trainable, "schedule": "a-fixed-budget",
                            "final_B": curve[-1]["Bval"], "final_A": curve[-1]["Aval"],
                            "final_P": curve[-1]["Pval"],
                            "eff_rank_mean": round(float(np.mean(ers)), 2), "curve": curve})
            print(f"  k={k}% r={r}: B {results[-1]['final_B']:.3f} A {results[-1]['final_A']:.3f} P {results[-1]['final_P']:.3f} effrank {results[-1]['eff_rank_mean']}")
    # frozen-base reference at each ladder point (the floor; adaptation must beat THIS)
    floors = {}
    for k in (0, 25, 50, 75, 100):
        model = build(cfg); model.load_state_dict(torch.load(OUT / f"ckpt_k{k}.pt", weights_only=True))
        floors[k] = {"B": evaluate(model, *evB), "A": evaluate(model, *evA), "P": evaluate(model, *evP)}
        print(f"  frozen k={k}%: B {floors[k]['B']:.3f} A {floors[k]['A']:.3f} P {floors[k]['P']:.3f}")
    (OUT / f"sweep_sched_a{'_full' if a.full else ''}.json").write_text(json.dumps(
        {"exploratory": True, "seed": SEED, "adapter_steps": a.adapter_steps, "lr": 5e-4,
         "floors_frozen": floors, "arms": results,
         "note": "schedule (a) fixed adapter budget; T-mid arms co-train the base on B (moving-base regime); k=100 is frozen-base (T-post). Label rows by regime."}, indent=1))
    print("sweep saved →", OUT / f"sweep_sched_a{'_full' if a.full else ''}.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("pretrain", "sweep"), required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--adapter-steps", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--full", action="store_true", help="sweep all ranks {2,8,32}")
    a = p.parse_args()
    (mode_pretrain if a.mode == "pretrain" else mode_sweep)(a)
