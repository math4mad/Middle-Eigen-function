# PREREG — H9-M: the critical window on MEF's byte-LM, measured as **gain over the same-k floor**, with a no-adapter control

**Registered:** 2026-09-13, machine A (`m1pro-32g`), by an MEF session named by the human's words
*"start H9 first"* — the Anatomist's hand, on the bench that owns the only rig in the programme where
**k** (injection time) actually moves.
**Frozen: 2026-09-13.** No number produced after this commit may be compared against a band, a
threshold or a referent that was written down later than it.
**Anchors:** draft consumed — `Kairos/docs/PREREG.md` §H9-M @`Kairos@ae6be01` (`sha256 5cdd05a74f3c2dd4…`,
**a draft, deliberately: Kairos owns the question and no apparatus**); travel clause — Letter 011 §2
(*"If no dose qualifies… the question travels to MEF's conditional checkpoint path or dies there, not
here"*), triggered by Step 0's measured no (`artifacts/results/sarcos/Step0_dose_*_B.json`,
`e3f7fa1955cf…`, `91c34b5047ec…`); R2's checkpoint condition — satisfied at
`chora:artifacts/checkpoints/stage18_kairos_ladder.json`; design evidence (NOT a result) —
`artifacts/results/mef/sweep_sched_a_full.json` (`bf2f74cfa61a…`, A seed 13) and
`…/E0_seed14_sweep_full.json` (`695dba48ff1d…`, B seed 14), **both `exploratory: true`**.

> **H9 (Sitting 003, tabled):** for fixed task and budget, the minimal LoRA rank r(k) achieving equal
> effect is monotone increasing in injection checkpoint k, superlinearly past some window.

**This registration changes the currency and keeps the claim.** Reading the two existing grids against
H9's own referent showed the referent is unusable on this rig — not wrong-valued, *unusable*:

| what the wording does here | measured from the pinned bytes |
|---|---|
| the base moves **5×** along the ladder | floor_B: 5.5456 → 1.4038 → 1.2194 → 1.1525 → 1.0906 (A seed 13); 5.6587 → … → 1.0313 (B seed 14) |
| so "minimum rank matching the k=0 run" is answered by the **base alone** at late k | k = 75 and 100 (A), k = 100 (B) already beat the k=0,r=8 target |
| and it is not stable even at its **reference rung** | r(0) = **2** on A, **8** on B — same protocol, one seed and one laptop apart |

A metric that turns a *closing* window into a *free* one cannot carry a monotonicity prediction, and a
metric whose value at k=0 moves with the laptop is measuring noise. So:

## 1 · Metric

All arms are MEF's stage18/19 byte-LM: 4 layers, d=192, nh=6, nkv=2, inter=512, ctx=256, 1.6 M params,
CPU/MPS `DEV`, AdamW, **adapter budget fixed at 600 steps at lr 5e-4 (schedule (a))**, stream B for
fitting, evaluation on the pinned held-out tensors `eval_B.pt` (primary), `eval_A.pt`, `eval_P.pt`
(probe, reported alongside, never scored). Base ladder: 2,000 pretrain steps, rungs latched at
k ∈ {0,25,50,75,100} % — `ckpt_k0.pt` is the **step-0 init** (verified in the code: it is saved before
the training loop runs).

* `floor_B(k)` — the frozen base evaluated **at** the rung, no further steps (already in the rig).
* `final_B(k, r)` — after LoRA at rank r plus 600 steps (**k<100: base co-trains — moving-base
  regime; k=100: base frozen — T-post**; two rows, never pooled).
* `ctrl_B(k)` — **new, amendment 1 below:** 600 further steps at the rung with **no adapter at all**.
* **`G(k,r) = floor_B(k) − final_B(k, r)`** — the draft's quantity, same-k reference.
* **`W(k,r) = ctrl_B(k) − final_B(k, r)`** — **what the adapter adds beyond 600 plain steps at the
  same rung.** This is the registered primary: the window is a statement about *plasticity*, and
  plasticity is what the control removes.

## 2 · Amendments the Anatomist makes to the draft (both against the rig, not against the question)

1. **A control row was missing, and without it the claim is confounded.** `G` decays along the ladder
   and so does *everything* — the floor at k=100 is already 1.09, near the entropy of the stream, so
   "600 steps buy less when you start late" is true of plain continued training and says nothing about
   LoRA. The registered primary is therefore **W(k,r) = ctrl_B(k) − final_B(k,r)**, with `G` reported
   beside it. Cost: 5 control arms ≈ 8 min on A's measured units. Apparatus: `CONTROLS=1` in
   `scripts/stage18_kairos_mini.py`, additive, default behaviour unchanged.
2. **The lr wrinkle, named rather than inherited.** E3's design finding was that a constant-lr rig
   cannot adjudicate a *schedule* signature (`e3_summary.json`, `finding_design_first`). H9-M does not
   claim one: every arm shares lr 5e-4 flat and a fixed 600-step budget, so **nothing in W may be
   attributed to a decaying schedule, because there is none** — the sentence is written here so the
   inheritance is not silently re-imported.

## 3 · The registered check (one hypothesis check, a conjunction; per-rung halves describe)

> **(H9-M)** with the band fixed as in §4, and for each rank r:
> **(i)** `W(k,r)` is strictly decreasing in k across {0,25,50,75,100} with every step beyond band;
> **(ii)** it is superlinear — the ratio `W(k_{i+1},r)/W(k_i,r)` falls below **½** at least once along
> the ladder;
> **(iii)** the window is nearly **rank-blind**: `W(k,2) ≥ 0.9·W(k,32)` at every rung.
> The conjunction is the check. H9-M **HOLDS** iff all three hold on the registered seeds.

**Secondary reading, cannot rescue or upgrade:** `r(k)` = min r with `W(k,r) ≥ ½·W(k,32)`, reported
because H9-S asked for it, scored never — §'s inversion table is why.

## 4 · Band — from k=0 replicates on **one machine**, frozen before any k>0 arm is compared to it

For each rank r, over the **A-machine** seeds present at k=0 (13, 14, and 15 when it exists):
`sd_r` = sample sd (ddof=1) of `W(0,r)`; **band = max over r of 2·sd_r**; minimum admissible replicate
count **n = 2**. B's seed-14 grid is a **cross-machine twin** — it closes Gate 6 and is reported
separately, and **may not be pooled into the band**, because a two-laptop spread confounds machine
with seed. The band is computed from k=0 arms only; once written to
`artifacts/results/mef/stage19_h9m/band.json` with the hash of every input, it is **not recomputed**
after any k>0 number is read. `h9m_seed15_and_band.py` prints candidates and refuses to name one;
naming is this commit's job and it is done above.

## 5 · Seeds, budget, and what the old bytes may be used for

* **Seeds 13, 14, 15 on A** (13 exists; 14 and 15 are this check's new work). B's seed 14 stays
  separate, as §4 says.
* **Budget, in units both machines have already printed:** base ladder 260.2 s (A) / 742.5 s (B);
  one arm 96.2 s (A). So per new seed: 260 s + 15 arms (≈ 24 min) + 5 controls (≈ 8 min) ≈ **35 min**;
  two seeds ≈ **1 h 10 m**. R1 densification (5-point rungs where |ΔW/Δk| exceeds band) only after
  §4's band exists — the registered guard is *"refining a bandless curve is fitting to noise with
  extra steps"*.
* **The two existing exploratory grids are design evidence and nothing else:** they motivated the
  currency change and they may be cited for that. They may **not** be scored — they predate this
  registration, they carry `exploratory: true`, they have no control row, and their rank ladders were
  never meant to answer H9. Registered runs write `exploratory: false` and name this commit via
  `REGISTERED=<sha>`.

## 6 · Obituary, written now

If **W is flat in k within band** — i.e. what you can buy with a rank-limited increment does not depend
on when you inject it, once you have subtracted what 600 plain steps would have bought anyway — then
**there is no critical window in this rig**, 设想5's metaphor dies on the bench that has the only k
ladder in the programme, and **the author of the dossier gets the obituary, loudly.** That sentence is
the Warden's, moved intact from Sarcos's leg because it was always the right sentence.
If instead W decays but §3(iii) fails — rank starts to matter at late k — then the capacity reading
returns and must come back as a **new registered claim (H9-M′)**; it may not be retro-fitted into this
one, and per R4 any post-training fit of these measures carries `post-hoc` forever.

*Signed: The Anatomist (MEF, `scripts/stage18_kairos_mini.py` @ this commit), for the seat; the question
belongs to The Horologist, the apparatus to this bench, and the venue was decided by Letter 011 §2 and
by nothing else.*
