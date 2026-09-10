# Stage 15 · 把唯一可能升级为主张的两个正向结果做实（seed 复验）

## 背景
本项目所有效应量级都被 stage9 压到 **+0.02~+0.03 = 一个 dev 噪声带（±0.030）**。
只有两个结果**同时**满足"方向为正、幅度超过噪声带、且是干预实验（不是观察）"：

| 结果 | 现有证据 | 缺什么 |
|---|---|---|
| **`ATOMfull`**（注意力+MLP 120 个矩阵，方向全冻结、只学 896×120=72,448 个原子系数） | best dev **0.578 vs 控制 0.556（+0.022）**，参数 1/61 | 只有 **1 个 seed**；且 dev_loss 更差（1.84 vs 1.13）⇒ 可能是校准/epoch 选择效应 |
| **`SHUFg`**（保谱全局重排：σ 多重集/能量/范数/秩全不变） | seed1 **−0.004**、seed2 待跑；带内 `SHUFb` −0.011/−0.025 | 需要 seed2/3 才能把"排序位置不携带功能信息"从"没测出差异"升级为"有把握的 null" |

这两个是明天唯一值得先做的：**它们决定我们下一步写进结论的是哪句话。**

## 假设
- **H15-1**：`ATOMfull` 的增益可复现 ⇒ 3 seed 配对 Δ 均值 > +0.02 且 ≥2/3 同号
  ⇒ 主张："**小样本微调的能力上限由自由度决定，而不是由'挑对大奇异值方向'决定**"
  （72k 标量、零新方向，追平/超过 4.4M 参数的 LoRA r=8）。
- **H15-2**：`SHUFg` 的 null 稳定 ⇒ 3 seed 配对 Δ 均值落在 ±0.030 内、95% CI 含 0
  ⇒ 主张："**在 MLP `down_proj` 上，σ 与方向的配对关系不携带下游功能信息**"
  （这条是"按大小切割欠妥"的最强正面证据）。
- **H15-3（对照）**：若 `ATOMfull` 复验翻车（均值 ≤0）⇒ 归为 stage5 式单 seed 假象，
  路线退回"只做监测"（stage10/17），不再谈"重参数化"。

## 方法（全部复用现有脚本，零新代码）
```bash
source .venv/bin/activate && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
QW05=$(pwd)/models/models/Qwen--Qwen2.5-0.5B/snapshots/master

# ① ATOMfull：补 seed 2、3（含自带 lora 控制组，同 seed 配对）
for s in 2 3; do
  MODEL_PATH=$QW05 SEEDS=$s TARGETS=both OUT_DIR=outputs/stage14_both \
    python -u scripts/stage14_rank1_atoms.py lora ATOMfull ATOM64 RAND64
done

# ② SHUFg / SHUFb：补 seed 3（seed1 已有，seed2 已在 stage11 跑过则跳过）
MODEL_PATH=$QW05 SEEDS=3 OUT_DIR=outputs/stage11 \
  python -u scripts/stage11_random_subspace.py lora SHUFb SHUFg SHUFinv
```
注意：`stage14_rank1_atoms.py` / `stage11_random_subspace.py` 都有 `results_partial.json` 续跑保护，
崩了直接重跑同一条命令即可；`DATA_SEED=42` 固定 ⇒ 与 stage9/10/11/14 全部可比。

## 统计规矩（写死，不许改）
1. **配对**：同 seed 的 `组 − lora控制`，不做跨 seed 平均后比较；
2. 同时报 **best-of-5** 与 **末 3 epoch 均值**（后者方差更小，stage9 已验证）；
3. 判读一律以 **±0.030 噪声带** 为准；3 seed 时给 t 检验 + bootstrap 95% CI（沿用
   `scripts/summarize_stage9.py` 里的 `paired_stats` 写法）；
4. 若某组 best 与 last3 结论相反 ⇒ 报"不稳定"，不许挑好看的那个。

## 成本与顺序
- ① 6 次运行 ≈ 30 分钟（MLP 大矩阵 SVD 使每次约 5–6 分钟）；② 4 次 ≈ 20 分钟。合计 **≈ 55 分钟**。
- 先跑 ①（决定主张），再跑 ②（null 的加固）。若时间只够一半，**优先 `ATOMfull`**。

## 产出
- `outputs/stage14_both/results.json`、`outputs/stage11/results.json`（含 seed 2/3）
- 报告页 `report/notes/atom-coordinates.qmd` §5/§3 的表会自动更新；
  §7 判决里把"单 seed"字样改成实测 seed 数与合并 Δ±CI。
