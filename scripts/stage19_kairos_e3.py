"""Stage 19 · KAIROS-E3 — the "last-mile" diagnostic (I-02's adjudication, born new).

DESIGN SOURCE: Kairos docs/IDEAS.md I-02 (signed by the human + Horologist):
fully-consolidated base, adapter gain ×10, log per-layer gradient norm +
Fisher spectrum + loss trajectory. Three mutually exclusive signatures:
  S-absorb  : gradient large & loss stuck        → soil truly occupied
  S-gate    : gradient small & directions healthy → throughput-gated (I-02)
  S-sched   : gradient large & loss falls         → it was only the lr schedule
If none fires, that is a result (the partial-improvement zone); reported,
never quietly dropped.

PRE-REGISTRATION (committed before any E3 number exists — this block is the
document; thresholds frozen here, not tuned there):

  Arms per seed s ∈ {13,14,15}: base = MiniQwen trained on A 600 steps exactly
  as stage18 mode_pretrain (k=100 point; seed-13 reuses stage18's ckpt_k100.pt
  if present — byte-identical base by construction, verified by hash);
  LoRA r=8 on all q,k,v,o,gate,up,down; base frozen (frozen-base-increment
  regime, T-post row of Letter 014 amendment 4); train adapter on stream B,
  600 steps, lr 5e-4 constant (same as stage18 schedule (a) — deliberately
  NO cosine, so "it was the schedule" cannot hide). Gain arms g ∈ {1,10}
  multiply the LoRA scaling s = alpha/r; identical init otherwise.

  Quantities (computed from the logged per-layer vectors):
    G_early = mean over layers of ||grad_layer|| averaged over steps 1..100
    G_late  = the same over steps 501..600
    decay   = G_late / G_early
    FR_late = Fisher effective rank (participation ratio of eigenvalues of
              the empirical Fisher G G^T / m, m=100 accumulated late-window
              per-layer grad vectors), averaged over layers
  Signatures (decided per seed; all three seeds must agree for the verdict
  to be claimed; disagreement = "not identified", filed either way):
    loss_stuck  : ΔBval = Bval(step 100) − Bval(final) < 0.010 nats
    loss_falls  : ΔBval ≥ 0.050 nats   (0.010–0.050 = partial zone, no claim)
    grad_large  : decay ≥ 0.50      grad_small: decay < 0.25  (0.25–0.50 = gray)
    dirs_healthy: FR_late(gain10) ≥ 0.75 × FR_late(gain1)   [same-base control
                  comparison; no absolute constant invented]
  S-absorb = grad_large & loss_stuck.  S-gate = grad_small & dirs_healthy.
  S-sched  = grad_large & loss_falls.  Overlap of S-absorb and S-sched is
  impossible (stuck vs falls); S-gate vs the others is separated by decay.

  KNOWN CONFOUND, declared, not hidden: AdamW is (nearly) invariant to a
  global rescaling of a parameter's gradient, so gain ×10 does not simply
  ×10 the step size; the gain arms differ in forward curvature they see
  (loss surface as a function of the adapter), not in nominal lr. The
  diagnostic therefore reads the CHANNEL (decay, FR, loss), not "more
  effective lr". If gain1 and gain10 are indistinguishable in all three
  quantities, the honest verdict is "gain is not a throttle knob under
  Adam" — filed as a finding about I-02's mechanism list.

  Bands across seeds: report sd of (ΔBval, decay, FR) across the three
  seeds next to everything. No test-set tuning anywhere; B is the fit
  stream, A/P logged as drift references only.

Usage (MEF root):
  .venv/bin/python scripts/stage19_kairos_e3.py --seed 13
  (loop seeds externally; one process per seed so env-pinned constants
   inside stage18_kairos_mini are honest)
"""
import argparse, json, os, sys, hashlib
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    os.environ["SEED"] = str(a.seed)
    os.environ["OUT_DIR"] = f"outputs/stage19_kairos_e3/seed{a.seed}"
    sys.path.insert(0, str(Path(__file__).parent))
    import numpy as np, torch, torch.nn.functional as F
    import stage18_kairos_mini as s18
    OUT = Path(os.environ["OUT_DIR"]); OUT.mkdir(parents=True, exist_ok=True)
    cfg = dict(d=192, layers=4, nh=6, nkv=2, inter=512, ctx=256)

    # ---- base at k=100 (consolidated): reuse stage18 seed-13 ckpt or train fresh
    k100 = Path("outputs/stage18_kairos_mini/ckpt_k100.pt")
    reuse = (a.seed == 13 and k100.exists())
    if reuse:
        meta = json.loads(Path("outputs/stage18_kairos_mini/base_run.json").read_text())
        base_sha = hashlib.sha256(k100.read_bytes()).hexdigest()
        steps, batch = meta["steps"], meta["batch"]
        print(f"base reused from stage18 seed13 (sha {base_sha[:12]})")
    else:
        A0, _, _, meta0 = s18.load_bytes()
        model = s18.build(cfg)
        rng = np.random.default_rng(a.seed)
        gen = s18.stream(A0, 16, cfg["ctx"], rng)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        import time; t0 = time.time()
        for step in range(600):
            x, y = next(gen); opt.zero_grad()
            loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        torch.save(model.state_dict(), OUT / "base_k100.pt")
        base_sha = hashlib.sha256((OUT / "base_k100.pt").read_bytes()).hexdigest()
        k100 = OUT / "base_k100.pt"; steps, batch = 600, 16
        print(f"fresh base trained ({time.time()-t0:.0f}s), sha {base_sha[:12]}")
    del steps, batch
    # ---- data + eval tensors (same packers as stage18; SEED env pins determinism)
    A, B, P, meta = s18.load_bytes()
    evA = torch.load("outputs/stage18_kairos_mini/eval_A.pt", weights_only=True) if reuse else None
    evB = torch.load("outputs/stage18_kairos_mini/eval_B.pt", weights_only=True) if reuse else None
    evP = torch.load("outputs/stage18_kairos_mini/eval_P.pt", weights_only=True) if reuse else None
    if evA is None:
        pack = s18.eval_tensors(A, B, P, cfg["ctx"]); evA, evB, evP = pack["A"], pack["B"], pack["P"]
        torch.save(evA, OUT / "eval_A.pt"); torch.save(evB, OUT / "eval_B.pt"); torch.save(evP, OUT / "eval_P.pt")

    arms = {}
    for gain in (1, 10):
        torch.manual_seed(a.seed)
        model = s18.build(cfg); model.load_state_dict(torch.load(k100, weights_only=True))
        s18.lora_targets(model, 8)
        for n, m in model.named_parameters():
            if ".A" not in n and ".B" not in n: m.requires_grad = False
        if gain != 1:
            for mod in model.modules():
                if isinstance(mod, s18.LoRA): mod.s = mod.s * gain
        adapter_ps = [p for n, p in model.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW(adapter_ps, lr=5e-4)
        rng = np.random.default_rng(a.seed + 7)          # same stream as stage18 sweep
        genB = s18.stream(B, 16, cfg["ctx"], rng)
        # per-layer bookkeeping: group adapter params by LoRA module (A and B of the same module)
        lora_mods = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, s18.LoRA)]
        rec = {name: {"grad_norms": [], "fisher": []} for name, _ in lora_mods}
        curve = []
        import time; t0 = time.time()
        for i in range(600):
            x, y = next(genB); opt.zero_grad()
            loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)); loss.backward()
            # logging BEFORE clip/step: the raw channel readout
            flat_by_mod = {}
            for name, mod in lora_mods:
                g = torch.cat([mod.A.grad.reshape(-1).float(), mod.B.grad.reshape(-1).float()])
                flat_by_mod[name] = g
                rec[name]["grad_norms"].append(float(g.norm()))
                if i >= 500:                                   # late window: 100 grad vectors
                    rec[name]["fisher"].append(g)
            torch.nn.utils.clip_grad_norm_(adapter_ps, 1.0); opt.step()
            if (i + 1) % 50 == 0:
                ev = {"Bval": s18.evaluate(model, *evB), "Aval": s18.evaluate(model, *evA),
                      "Pval": s18.evaluate(model, *evP)}
                curve.append({"step": i + 1, "loss": float(loss), **ev,
                              "gnorm_mean_layers": float(np.mean([rec[n]["grad_norms"][-1] for n, _ in lora_mods])),
                              "secs": round(time.time() - t0, 1)})
                print(f"seed{a.seed} g{gain} step {i+1} B {ev['Bval']:.3f} A {ev['Aval']:.3f} P {ev['Pval']:.3f}")
        # Fisher effective rank per layer from the accumulated late gradients
        frs = {}
        for name, _ in lora_mods:
            Gm = torch.stack(rec[name]["fisher"]).T            # d × m
            K = (Gm.T @ Gm) / Gm.shape[1]                      # m × m gram == nonzero eig of Fisher
            evs = torch.linalg.eigvalsh(K).clamp_min(0)
            frs[name] = float((evs.sum() ** 2) / (evs ** 2).sum())
        gnorm_early = {name: float(np.mean(rec[name]["grad_norms"][:100])) for name, _ in lora_mods}
        gnorm_late = {name: float(np.mean(rec[name]["grad_norms"][500:])) for name, _ in lora_mods}
        arms[gain] = {
            "curve": curve, "fisher_effrank_per_layer": frs,
            "grad_norm_early_per_layer": gnorm_early, "grad_norm_late_per_layer": gnorm_late,
            "decay_per_layer": {k: gnorm_late[k] / max(gnorm_early[k], 1e-12) for k in gnorm_early},
            "decay_mean": float(np.mean(list(gnorm_late.values())) / max(float(np.mean(list(gnorm_early.values()))), 1e-12)),
            "FR_mean": float(np.mean(list(frs.values()))),
            "dBval_100_to_end": curve[1]["Bval"] - curve[-1]["Bval"],
        }
    # ---- adjudication (thresholds frozen in this file's header)
    d10, d1 = arms[10]["decay_mean"], arms[1]["decay_mean"]
    fr10, fr1 = arms[10]["FR_mean"], arms[1]["FR_mean"]
    dl = arms[10]["dBval_100_to_end"]
    grad_large, grad_small = d10 >= 0.50, d10 < 0.25
    stuck, falls = dl < 0.010, dl >= 0.050
    healthy = fr10 >= 0.75 * fr1
    fired = [n for n, b in {"S-absorb": grad_large and stuck,
                            "S-gate": grad_small and healthy,
                            "S-sched": grad_large and falls}.items() if b]
    out = {"exploratory": False, "seed": a.seed, "base_sha256": base_sha, "corpus": meta,
           "gain_arms": {str(g): arms[g] for g in arms},
           "quantities_used": {"gain10": {"decay": d10, "FR": fr10, "dBval": dl},
                               "gain1": {"decay": d1, "FR": fr1}},
           "thresholds": "see header PRE-REGISTRATION block (frozen at this commit)",
           "fired": fired,
           "verdict_seed": fired[0] if len(fired) == 1 else ("AMBIGUOUS:" + "+".join(fired) if fired else "NONE (gray/partial zone)"),
           "run_on": {"host": s18.socket.gethostname(),
                      "machine": os.environ.get("CHORA_MACHINE", s18.platform.machine()),
                      "torch": s18.torch.__version__, "py": s18.platform.python_version()}}
    (OUT / "e3_verdict.json").write_text(json.dumps(out, indent=1))
    print(f"seed {a.seed} → {out['verdict_seed']}  (decay {d10:.3f}, FR {fr10:.2f} vs {fr1:.2f}, ΔBval {dl:.3f})")

if __name__ == "__main__":
    main()
