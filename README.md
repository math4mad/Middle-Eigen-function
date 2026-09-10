
>  Original Thoughts is here :
   - [泛函空间再向上抽象，然后呢？](https://math4mad.github.io/talk-with-agents/talking/mathematics/functional-spaces-to-singular-values.html)
   - [4. 神经网络的参数矩阵，能像普通矩阵一样 SVD 压缩吗？](https://math4mad.github.io/talk-with-agents/talking/mathematics/functional-spaces-to-singular-values.html)
   - [8. 不同公司的模型，rank-1 主奇异值矩阵相似吗？](https://math4mad.github.io/talk-with-agents/talking/mathematics/functional-spaces-to-singular-values.html)
   - [11. 从低维到高维的「生成」，就是人类的学习机理](https://math4mad.github.io/talk-with-agents/talking/mathematics/functional-spaces-to-singular-values.html)
   - [13. 改动分解后的某些特征值，有可能、有意义吗？](https://math4mad.github.io/talk-with-agents/talking/mathematics/functional-spaces-to-singular-values.html)
   - [14. 大奇异值是泛化，最小奇异值是噪音，中间那一组是什么？](https://math4mad.github.io/talk-with-agents/talking/mathematics/functional-spaces-to-singular-values.html)

   -  [编者注 · Editor’s note：把「中间那一组」画出来](https://math4mad.github.io/talk-with-agents/talking/mathematics/functional-spaces-to-singular-values.html)
   -  补充更新（我大脑网络的自适应微调）：这里经过几天大脑的自适应， 可能有些概念又更新了。  SVD分解的特征值按常规操作，从大到小排序。 但是综合数据是混合部分。所以考虑大小来切割是欠妥的。 但是整套流程已经为我们随机抽取特征矩阵构建子学习空间铺垫了技术基础

   --- 

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
