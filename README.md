# Word2Vec 词向量训练与可视化项目

## 项目简介

本项目演示了如何使用 Python 和 Gensim 库训练 Word2Vec 词向量模型。项目包含从文本预处理、模型训练到结果可视化的完整流程。特别针对中文环境集成了 Jieba 分词，并通过 t-SNE 算法将高维词向量降维至 2D 平面，直观展示词语间的语义分布关系。

## 功能特点

- 集成 Jieba 分词与自动停用词过滤。
- 支持 CBOW 或 Skip-gram 架构训练。
- 自动保存训练好的模型 (`*.model`)。
- 生成词向量分布散点图，可视化词语相似度。

## 环境要求

- Python 3.8+
- 见 `requirements.txt`

## 项目结构

```
word2vec_project/
│
├── README.md              # 项目说明文档
├── requirements.txt       # 项目依赖库
├── data/
│   └── corpus.txt         # 训练语料数据（示例数据）
├── src/
│   ├── train.py           # 训练脚本（含数据清洗）
│   └── visualize.py       # 可视化脚本（生成图片）
└── models/
    └── word2vec.model     # (运行后生成) 训练好的模型文件
```

## 快速开始

### 1. 安装依赖

```
pip install -r requirements.txt
```

### 2. 准备数据

默认数据位于 `data/corpus.txt`。你可以直接运行使用内置示例，或替换为自己的中文文本文件。

### 3. 训练模型

在项目根目录运行：

```
python src/train.py
```

程序将自动分词、训练并保存模型至 `models/` 目录。

### 4. 可视化结果

训练完成后运行：

```
python src/visualize.py
```

生成的散点图将展示词语间的聚类关系。

## 常见问题

### 报错“未找到测试词汇”

- 原因：语料库太小，词汇被 `min_count` 过滤。
- 解决：确保 `train.py` 中 `min_count=1`。

### 相似词全是标点符号

- 原因：标点出现频率过高。
- 解决：代码已内置 `STOPWORDS` 自动过滤，若仍有干扰请在代码中补充标点。

### 找不到模型文件

- 解决：请确保在**项目根目录**运行命令，或查看控制台输出的绝对路径。

