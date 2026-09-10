#!/usr/bin/env bash
# Stage 9 runner — robustness checks for the L6 ("deep top-spectrum = harmful memory") finding.
# See stage/stage9.md (hypotheses / criteria) and stage/NEXT.md (command list).
# Phases run sequentially; a phase whose results.json already exists is skipped, so this
# script is re-launchable after an MPS crash.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p log

QW05="$(pwd)/models/models/Qwen--Qwen2.5-0.5B/snapshots/master"
QW15="$(pwd)/models/models/Qwen--Qwen2.5-1.5B/snapshots/master"

# run_phase <name> <model> <seed> <data_seed> <frac> <batch> <outdir> <groups...>
run_phase () {
  local name="$1" model="$2" seed="$3" dseed="$4" frac="$5" batch="$6" out="$7"; shift 7
  if [ -f "$out/results.json" ]; then
    echo "[skip] $name already finished ($out/results.json) $(date +%H:%M:%S)" | tee -a log/stage9.log
    return 0
  fi
  echo "[start] $name -> $out groups: $* (model=$(basename "$model") seed=$seed data_seed=$dseed frac=$frac batch=$batch) $(date +%H:%M:%S)" | tee -a log/stage9.log
  MODEL_PATH="$model" SEED="$seed" DATA_SEED="$dseed" FRAC_TRAIN="$frac" BATCH="$batch" \
    OUT_DIR="$out" python -u scripts/stage3_llm_lora.py "$@" > "log/stage9_${name}.log" 2>&1
  local rc=$?
  echo "[done ] $name exit=$rc $(date +%H:%M:%S)" | tee -a log/stage9.log
}

for ph in ${PHASES:-seeds boundary scale data}; do
  case $ph in
    seeds)     # ① 多 seed 复验：数据子集固定(DATA_SEED=42)，只变训练种子
      for s in 1 2 3; do
        run_phase "seed$s" "$QW05" $s 42 0.10 16 "outputs/stage9_seed$s" baseline TAIL6
      done ;;
    boundary)  # ② 边界细化：TAIL3 / TAIL6 / TAIL12 的倒 U
      run_phase boundary "$QW05" 42 42 0.10 16 "outputs/stage9_boundary" TAIL3 TAIL6 TAIL12 ;;
    scale)     # ④ 规模检验：Qwen2.5-1.5B（28 层 → TAIL7 = 后 25% 深度，TAIL14 为过头对照）
      run_phase 1p5b "$QW15" 42 42 0.10 8 "outputs/stage9_1p5b" baseline TAIL7 TAIL14 ;;
    data)      # ③ 数据规模 10%→30%（新数据子集 DATA_SEED=7）
      run_phase 30pct "$QW05" 42 7 0.30 16 "outputs/stage9_30pct" baseline TAIL6 ;;
    *) echo "unknown phase: $ph"; exit 1 ;;
  esac
done
echo "ALL PHASES DONE $(date)" | tee -a log/stage9.log
