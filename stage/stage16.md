# Stage 16 · 把"能量"这个混淆因子从 Stage 12 的判决里彻底剥掉

## 背景（stage12 留下的唯一漏洞）
Stage 12 在同一污染流上删**同样条数**（K=2×24 层）的原子，三种坐标的 best dev（f=0.5、seed1、noop=0.527）：

| 探针 | TGT | RAND | SPEC |
|---|---|---|---|
| 混合指纹（f=0.5 流） | 0.502 (−0.025) | 0.516 (−0.011) | 0.523 (−0.004) |
| 纯垃圾指纹（f=1.0 流） | 0.495 (−0.032) | 0.516 (−0.011) | 0.523 (−0.004) |

⇒ 判决"三组全 ≤0、TGT 最差 ⇒ 撤回删除式手术"。
**但能量没配平**：TGT 每层削掉 **0.37%**、RAND **0.25%**、SPEC **0.19%**（因为 TGT 命中的原子
σ 更大）。所以"TGT 比 RAND 差 0.022"里有一部分是"多削了 0.12% 能量"，不是"取向选错"。
本阶段用**等能量选择**把这个漏洞堵死 —— 否则这条判决是可以被反驳的。

## 假设
- **H16-1（主）**：等能量后 `TGT` 仍不优于 `RAND`/`SPEC`（Δ 在 ±0.030 内）
  ⇒ **污染在方向层面不可与任务学习分离**，删除式手术正式关闭。
- **H16-2**：等能量后 `TGT` 显著优于 `RAND`（>+0.03）
  ⇒ 推翻 stage12 的判决：**污染可定点清除**，则立刻开 Stage 16b（把 TGT 删除做成在线干预：
  监测到 `mid_E`/`rel` 超阈 → 实时删除对应原子 → 看能否止住损伤），这是"抗污染训练"的起点。
- **H16-3（对照）**：`SPEC` 与 `RAND` 等能量后仍打平
  ⇒ "按谱位切"≈"随机切"（与 stage5 等能量 +0.015、stage13 τ≈0、stage14 ATOM≈RAND 四条证据闭合）。

## 方法（改 30 行代码，不动协议）
在 `scripts/stage12_targeted_deletion.py` 里加两个能力：

1. **等能量选择器**（核心）：给定目标能量 $E^\star$（= TGT 实际削掉的 $\sum\sigma_i^2$），
   用**拒绝采样**或**贪心**选出原子集合 $S$ 使 $\left|\sum_{i\in S}\sigma_i^2-E^\star\right|/E^\star\le5\%$，
   并支持三种取向约束：
   - `EM-TGT`：TGT 本身（基准）
   - `EM-RAND`：全谱随机（等能量、条数可变）
   - `EM-BAND`：**同分位带内**随机 + 等能量（比现在的 RAND 更严格）
   - `EM-SPEC`：6–10 带内选到等能量为止
   实现要点：`choose_masks(..., energy_target=…, tol=0.05)`，返回 `(masks, achieved_energy)`；
   并把每组实际能量写进结果（现在已有 `energy_removed_mean`，再加 `energy_target` 与 `energy_err`）。
2. **剂量-反应网格**（回答 stage10.md 原始问题"截取的数量 vs 注入比例是否相关"）：
   `K ∈ {1,2,4,8}` × `f ∈ {0.25,0.5}` × `EM-TGT`，seed 1–2；
   输出"回收/损伤 vs (K,f)"的曲面表 + 相关系数。

```bash
QW05=$(pwd)/models/models/Qwen--Qwen2.5-0.5B/snapshots/master
# ① 等能量三组（f=0.5，seed1–2）
MODEL_PATH=$QW05 SEEDS=1,2 JUNK_FRACS=0.5 PROBE_F=1.0 ARMS=tgt,emrand,emband,spec K=2 \
  OUT_DIR=outputs/stage16 python -u scripts/stage12_targeted_deletion.py
# ② 剂量网格（EM-TGT）
MODEL_PATH=$QW05 SEEDS=1,2 JUNK_FRACS=0.25,0.5 PROBE_F=1.0 ARMS=tgt K=1,4,8 \
  OUT_DIR=outputs/stage16_grid python -u scripts/stage12_targeted_deletion.py
```
（`K` 需从单值改为列表：`KS = [int(x) for x in os.environ.get("K","2").split(",")]`，
 tag 里带上 `K{k}`。）

## 统计规矩
- 与 stage12 同：同 seed、同流、同探针 ⇒ 配对；
- **必须报每组实际削掉的能量**（均值 ± 跨层标准差），并检验 TGT/EM-RAND 的能量差 ≤5%；
- 3 seed（stage15 之后可复用其控制组）；判读 ±0.030；
- 若能量无法在容差内配平（比如 TGT 命中极大 σ），改做**能量对齐到更小值**（都削到 min 能量）
  并在报告里写明选择规则。

## 成本
- ① 4 组 × 2 seed = 8 次运行 + 2 次探针 ≈ **50 分钟**
- ② 3 个 K × 2 个 f × 2 个 seed = 12 次 ≈ **60 分钟**（探针可复用 ①）
- 合计 ≈ 1.8 小时；**只做 ① 也能出判决**（①优先）。

## 风险
- 删 2 个原子只削 0.2–0.4% 能量，本来就在噪声边缘 ⇒ 若等能量后仍全为负，
  结论强度受限于"干预太弱"；对策：加一组 `K=32`（削 ~5% 能量）做"强干预"上界，
  看损伤是否随 K 单调加深（若是，则"删不动"不是"删太轻"造成的）。
