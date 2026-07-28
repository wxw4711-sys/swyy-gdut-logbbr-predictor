---
title: logBBR 骨血比预测系统
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
license: mit
---

# logBBR 骨血比预测系统 (LightGBM QSAR)

基于 LightGBM 的 logBBR（骨/血分配比）预测网页应用，使用 Mordred 分子描述符 +
Morgan 指纹 + 实验元数据（体重/性别/剂量）作为特征。

## 公网地址（Hugging Face Spaces 免费层）
https://swyy-gdut-logbbr-predictor.hf.space

## 功能
- 单分子 / 批量 SMILES 预测 logBBR 与骨-血分配系数 Kp（=10^logBBR）
- 分子结构可视化与基于 Mordred 理化维度的活性区域溯源
- 模型特征重要性展示
- 批量预测支持附实测 logBBR，自动计算 Spearman ρ / Pearson r

## 本地运行
```bash
pip install -r requirements.txt
python app.py          # 启动 Gradio 界面（默认 http://127.0.0.1:7860）
# python flask_server.py   # 如需原 Flask 版本
```

## 输入格式（批量预测）
每行：`SMILES<TAB>体重(g)<TAB>性别<TAB>剂量<TAB>实测logBBR(可选)`
例如：`NS(=O)(=O)c1ccc(NC(=O)Nc2ccc(F)cc2)cc1<TAB>33.1<TAB>male<TAB>100<TAB>-0.74`
