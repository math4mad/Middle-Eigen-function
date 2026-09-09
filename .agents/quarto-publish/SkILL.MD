---
name: quarto-publish
description: >-
  Assists with building and deploy quarto website to  gh-pages
license: Apache-2.0
compatibility: "Requires Python 3.8+"
metadata:
  author: math4mad
  version: "1.0.0"
  category: data-ai
  tags: ["github", "machine-learning", "gh-pages", "quarto-template", "pipeline"]
---

# quarto 

## Overview

quarto is a scientific  note systemd  

## Instructions

-  when  get instruction , ref :https://github.com/wiesehahn/quarto_website_template
Quarto Website Template to create a empty framework of website

## 国内镜像与下载通道

GitHub / HuggingFace 直连不可用时，使用以下镜像：

- **Quarto CLI**：从 GitHub Releases 下载，可走加速代理，例如：
  ```bash
  # macOS pkg 例（版本号按需替换）；代理站可选 https://ghfast.top/ 、https://gh-proxy.com/ 、https://ghproxy.net/
  curl -sL -o quarto.pkg "https://ghfast.top/https://github.com/quarto-dev/quarto-cli/releases/download/v1.10.18/quarto-1.10.18-macos.pkg"
  # 免 sudo 安装：解开 pkg 后把 Payload 拷到 ~/.local/quarto，再将 ~/.local/quarto/bin 加入 PATH
  pkgutil --expand-full quarto.pkg expanded && cp -R expanded/quarto-core.pkg/Payload ~/.local/quarto
  ```
- **gh-pages 发布**：网络受限时改由 GitHub Actions 渲染发布（本仓库 `.github/workflows/publish.yml`）；
  `git push` 若报 SSL 证书错误，用 `GIT_SSL_NO_VERIFY=1 git push`。

##. Request
**User request:** "Create a new Quarto website with template repo"

