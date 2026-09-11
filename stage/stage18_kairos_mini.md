# Stage 18 · KAIROS-MINI — LoRA injection timing on a from-scratch small LM

**Status: EXPLORATORY.** This stage produces *calibration*, not verdicts: no
hypothesis is tested here, no table line shared with any registered
experiment (Letters 009/011 fences inherited whole). The registered sibling
of this question at MLP scale is Sarcos's Step 0 → H9-S; the number-exp9
itself remains proposed-reserved pending JacobiGP's NEXT.md.

## Why from scratch (the user's correction, in one line)

A pretrained 0.5B is a *final state*: T-pre and T-mid have no k to attach
to. Injection timing needs a base **lifetime we own** — checkpoints latched
at true percentages of our own schedule.

## Rig

- Architecture: Qwen2.5-0.5B's own family, explore scale — 4 layers, d=192,
  GQA 6q/2kv heads hd=64, SwiGLU inter 512, RMSNorm, RoPE θ=100 on ctx 256,
  tied head, **byte-level vocab 256**; Qwen's `initializer_range 0.02` on
  embeddings (std=1 puts untrained CE at ~176 nats, not ln 256 ≈ 5.5 —
  the k=0 floor is a measured quantity, it gets honest init). ≈ 1.6M params.
- Corpus: shared-store `data/tiny_stories.txt` (pinned 400 MB prefix of the
  TinyStories train stream, hf-mirror, MIT) — 95% = pretrain stream A;
  held-out tail = **B, the downstream stream** (true shift at this scale:
  unseen story text); case-toggled B = **P, the shift probe** (deterministic,
  derived, not stored).
- Ladder: base trained 2000 steps, b16, AdamW 3e-3, grad-clip 1.0, seed 13
  (the canonical Sarcos-family seed), checkpoints latched at k ∈ {0, 25, 50,
  75, 100}% of steps.
- Arms (schedule **(a): fixed adapter budget** only in this run): from each
  ladder point, inject LoRA (r ∈ {2, 8, 32}, α=32, all q/k/v/o/gate/up/down)
  and train **on B for 600 steps** — at k<100 the base keeps moving (that is
  the *training-under-constraint* regime, its own rows); at k=100 the base
  is frozen (*frozen-base*, the classical LoRA regime). Controls: frozen
  floor at every k, measured not assumed — the **floor-effect fence**
  (Letter 009): an arm's improvement is scored against its own k's floor.
- Metrics: B-NLL (downstream), A-NLL (in-distribution retention =
  forgetting), P-NLL (shift sensitivity), effective rank of merged ΔW per
  arm (participation ratio of σ(ΔW) across tagged modules), full curves.

## The questions this exploration is allowed to ask (none are registered)

1. **Is there gain at all?** Does any (k, r) beat its frozen floor on B by
   more than noise? (The Sarcos lesson: assume no until shown.)
2. **r(k) shape at LM scale:** for fixed final-B targets, does minimum
   qualifying rank rise with k — the H9 direction, at 1.6M params?
3. **Forgetting asymmetry:** do moving-base arms (k<100) pay A-Price the
   frozen arm (k=100) cannot (it cannot forget what it never learns into)?
4. **Effective-rank behaviour of ΔW when the base is still moving** —
   feeding MEF's own isospectrality audit debt with machine-generated
   within-bench data (never external hearsay).

## Known limits, stated before anyone cites them

Schedule (b) train-to-end not run yet; single seed (bands impossible — all
"improvements" here are descriptive until replicates exist); B's shift is
"unseen text" (weak at this scale — A/P bound the sensitivity story);
byte-level LM ≠ language (this is a timing apparatus, not a LM claim);
ctx 256 ignores the real 0.5B's 32k rope geometry. **None of these numbers
leave this directory except as "stage18 exploratory" with the run log's
sha256 attached.**

## Run log

`log/stage18_run1.log` — pretrain 2000 + sweep 15 arms + 5 floors
(~1 h on M1, measured ~0.3 s/step at b16 ctx256).
