# Middle Eigen Function — SVD 消融实验

验证假设：*预训练模型权重矩阵的中间奇异值（"松散谱"）是泛化能力的主要来源。*

- 任务书：[`AGETNTS.md`](AGETNTS.md)
- 实验代码：`scripts/stage2_svd.py`（谱分析）、`scripts/stage3_ablate_train.py`（A/B/C 消融 + RTE 小样本微调）
- 结果数据与图：`outputs/`
- 日志记录: `log/`
- 阶段任务:`stage/`
- Quarto 报告：`report/`（GitHub Actions 自动发布到 gh-pages）

## 快速复现

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 模型与数据（HuggingFace 不可达时用 ModelScope / OSS 镜像）：
python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/bert-base-uncased', cache_dir='./models')"
curl -sL -o data/RTE.zip "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/glue/RTE.zip" && (cd data && unzip RTE.zip)
python scripts/stage2_svd.py
python scripts/stage3_ablate_train.py baseline A B C
cd report && quarto render
```
