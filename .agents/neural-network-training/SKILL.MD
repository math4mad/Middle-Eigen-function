---
name: neural-network-training
description: >-
  training neural network 
license: Apache-2.0
compatibility: "Requires Python 3.8+"
metadata:
  author: math4mad
  version: "1.0.0"
  category: data-ai
  tags: ["neural network","paramters matrix"]
---



## Overview

general method of traing neural network by pytorch

## Instructions

  - follow : https://docs.pytorch.org/docs/2.14/index.html
  - based on  M1 pro MacBook pro  16 inch  10 core cpu , 16 core gpu , 32+512   select a  moderate   BERT series model  as our expreiment model 
  - pytorch using  mps  

## 模型与数据集：国内镜像

**模型**（HuggingFace 直连不通时，用 ModelScope）：

```python
from modelscope import snapshot_download
path = snapshot_download("AI-ModelScope/bert-base-uncased", cache_dir="./models",
                         allow_patterns=["*.json", "*.txt", "model.safetensors"])
model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)  # 本地路径加载
```

微调建议显式 `dtype=torch.float32`（transformers v5 部分环境默认 bf16，会影响 SVD 重构精度）。

**GLUE/RTE 数据集**（`load_dataset("glue","rte")` 需要连 HF Hub，不通时直接取 ModelScope 官方 OSS 镜像）：

```bash
mkdir -p data && cd data
curl -sL -o RTE.zip "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/glue/RTE.zip" && unzip RTE.zip
# 得到 RTE/train.tsv, RTE/dev.tsv（label 为 entailment/not_entailment 字符串，注意有1条脏数据 label 缺失需 dropna）
```

GLUE 其余子任务在同一 OSS 目录下：MNLI.zip / CoLA.zip / SST-2.zip / QNLIv2.zip / QQP-clean.zip / STS-B.zip。

**其它备用通道**：`api.github.com`、`raw.githubusercontent.com`、阿里云 OSS、ModelScope CDN 实测可达；
`github.com` 网页端走加速代理（见 quarto-publish）；pip 包默认可达 PyPI。

  
