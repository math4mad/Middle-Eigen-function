---
name: matrix-svd-decompostion 
description: >-
  excute matrix   dempcompostion for piplne operation 
license: Apache-2.0
compatibility: "Requires Python 3.8+"
metadata:
  author: math4mad
  version: "1.0.0"
  category: math 
  tags: ["matrix", "linear algebra", "svd", "rank-one","full-rank"]
---

# quarto 

## Overview

using SVD get  components of  Data Matrix

## 模型下载：国内镜像（HuggingFace 直连不通时）

用 **ModelScope** 代替 huggingface_hub 下载预训练权重：

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
# bert-base-uncased 在 ModelScope 上的镜像仓库 id：AI-ModelScope/<name>
p = snapshot_download("AI-ModelScope/bert-base-uncased", cache_dir="./models")
```

之后直接用本地路径加载即可（transformers 的 `from_pretrained(p)`）：

- 常用前缀：`AI-ModelScope/`（英文模型镜像）、`damo/`、`iic/`（达摩系中文模型）。
- 设 `HF_HUB_OFFLINE=1` 避免 transformers 去连 huggingface.co 重试超时。
- 注意：snapshot_download 会连 coreml/onnx/rust_model.ot 等冗余文件拖慢速度，
  可加 `allow_patterns=["*.json", "*.txt", "model.safetensors"]` 只取必需文件。



