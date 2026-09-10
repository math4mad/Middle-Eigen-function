# NEXT（下一批要跑）— Stage 15 / 16 / 17

| 阶段 | 任务书 | 一句话 | 最小可发版 | 预计 |
|---|---|---|---|---|
| **15** | [`stage15.md`](stage15.md) | 唯一两个可能升级为主张的正向结果做 seed 复验：`ATOMfull`(+0.022, 1/61 参数) 与 `SHUFg`(保谱重排的 null) | 只跑 `ATOMfull` seed2/3 ≈ 20 分钟 | 55 分钟 |
| **16** | [`stage16.md`](stage16.md) | 把"能量"混淆从 Stage 12 判决里剥掉：等能量选择器 + K×f 剂量网格（回答"截取数量 vs 注入比例是否相关"） | 只做等能量三组 ≈ 50 分钟 | 1.8 小时 |
| **17** | [`stage17.md`](stage17.md) | 把在线报警做实：5 条基线轨迹 + `rel`/`mid_E` 联合统计量 + 置换检验阈值（+ 梯度谱可选） | ①+② 纯离线 ≈ 50 分钟 | 2 小时 |

优先级：**15 → 16 → 17**（15 决定我们能把哪句话写进结论；16 决定 Stage 12 的判决能不能站住；
17 决定"抗污染训练"这条线是否继续）。统计规矩统一：同 seed 配对、±0.030 噪声带、
best-of-5 与末 3 epoch 均值同时报、崩了用 `results_partial.json` 续跑。

---

## （历史）Stage 9 清单 — L6 现象的三重检验【已跑完，结果见 report/notes/l6-robustness.qmd】

> 任务书见 [`stage9.md`](stage9.md)（假设/判据/生物类比/可行性）；本文件只放执行清单。

所有命令在项目根目录、`source .venv/bin/activate` 后执行。
脚本 `scripts/stage3_llm_lora.py` 已支持：`SEED`（训练种子）、`DATA_SEED`（固定=42 保证数据可比）、
`FRAC_TRAIN`（数据比例）、层组名 `TAILn/FRONTn/MIDn`（层数自适应，0.5B=24 层、1.5B=28 层）。

先设：
```bash
export HF_HUB_OFFLINE=1
QW05=$(pwd)/models/models/Qwen--Qwen2.5-0.5B/snapshots/master
QW15=$(pwd)/models/models/Qwen--Qwen2.5-1.5B/snapshots/master   # 若未下完: python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B', cache_dir='./models', allow_patterns=['*.json','*.txt','model*.safetensors*','merges.txt','vocab*'])"
```

## 1. L6 多 seed 复验（0.5B，数据不变只变训练种子）
```bash
for s in 1 2 3; do
  MODEL_PATH=$QW05 SEED=$s OUT_DIR=outputs/stage8_seed$s \
    python scripts/stage3_llm_lora.py baseline TAIL6
done
# 判据：TAIL6−baseline 的 dev 差在 3/4 个 seed 上同号且均值 > +0.03 → L6 效应成立
```

## 2. 边界细化：TAIL3 / TAIL6 / TAIL12
```bash
MODEL_PATH=$QW05 OUT_DIR=outputs/stage8_boundary \
  python scripts/stage3_llm_lora.py TAIL3 TAIL12
# 预期若"记忆在深层"：TAIL3≈TAIL6 > TAIL12（切过头伤中层骨架）
```

## 3. 数据规模演化：10% → 30%
```bash
MODEL_PATH=$QW05 FRAC_TRAIN=0.3 DATA_SEED=7 OUT_DIR=outputs/stage8_30pct \
  python scripts/stage3_llm_lora.py baseline TAIL6 TAIL12
# 判据：30% 数据下 TAIL6 优势若缩小 → 深层骨架"随训练变有用"（记忆→能力的转化）
```

## 4. 规模检验：Qwen2.5-1.5B（28 层）
```bash
MODEL_PATH=$QW15 BATCH=8 OUT_DIR=outputs/stage8_1p5b \
  python scripts/stage3_llm_lora.py baseline TAIL7 TAIL14
# 28 层取 25% 深度 = TAIL7；TAIL14 为过头对照；BATCH 减半防显存压力
```

## 5. 汇总 & 发布
```bash
python scripts/summarize_stage8.py   # 待写：读 stage8_* 目录出表+图
cd report && quarto render
git add -A && git commit -m "stage8: L6 replication" && git push
```

## 备注
- 单组时长参考：0.5B ≈ 12-18 分钟；1.5B 预计 25-35 分钟。全套 ≈ 3-3.5 小时。
- 若某组 MPS 崩溃（历史偶发），直接重跑该组即可（results_partial.json 会续用已有组）。
- 明天结果出来后：更新 `report/notes/qwen-deep.qmd` 第 6 节（把"限定"改写成"复验结论"）。
