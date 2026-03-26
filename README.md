# -word2vec-
利用word2vec来训练词向量

本项目是一个基于 Python 的完整 Word2Vec 演示流程，涵盖了从中文语料预处理、模型训练到高维向量可视化的全过程。

📂 项目结构
text
word2vec_project/
│
├── README.md              # 项目说明文档（本文件）
├── requirements.txt       # 项目依赖库
├── data/
│   └── corpus.txt         # 训练语料（中文文本）
├── src/
│   ├── train.py           # 训练脚本（含 Jieba 分词与清洗）
│   └── visualize.py       # 可视化脚本（t-SNE 降维绘图）
├── models/                # (运行后生成) 存放训练好的模型文件
│   └── word2vec.model
└── results/               # (运行后生成) 存放可视化图片
    └── word_vectors_plot.png

🛠️ 环境准备
本项目依赖于以下 Python 库，请确保已安装：

Python 3.8+
Gensim: 用于训练 Word2Vec 模型。
Jieba: 用于中文分词（核心组件）。
Matplotlib & Scikit-learn: 用于绘图和 t-SNE 降维。

安装依赖

在终端中运行以下命令：

bash
pip install -r requirements.txt

🚀 快速开始
准备数据
默认数据文件为 data/corpus.txt。
你可以直接运行，使用内置的示例数据。
若要使用自己的数据，请将文本放入该文件，建议每行一句话或一个段落。

训练模型
在项目根目录下运行训练脚本：
python src/train.py

脚本将执行以下操作：
读取 data/corpus.txt。
使用 Jieba 进行分词，并自动过滤标点符号和停用词。
训练 Word2Vec 模型（Skip-gram 模式）。
将模型保存至 models/word2vec.model。
打印与“人工智能”最相似的词以验证效果。

可视化结果
训练完成后，运行可视化脚本：
python src/visualize.py

脚本将执行以下操作：
加载训练好的模型。
选取 10 个核心词汇（如“人工智能”、“自然”、“数据”等）。
使用 t-SNE 算法将 100 维向量降维至 2D 平面。
生成散点图并保存至 results/word_vectors_plot.png。

⚙️ 核心参数说明
在 src/train.py 中，你可以根据数据量调整以下参数：
参数   默认值   说明
min_count   1   词频阈值。忽略出现次数少于该值的词。小语料库请务必设为 1。

epochs   50   迭代次数。小语料库需要更多迭代次数（如 20-50），默认值通常太小。

window   2   上下文窗口。小语料库建议设小一点（如 2-5），以捕捉紧密的语义关系。

sg   1   算法选择。1 为 Skip-gram（适合小数据），0 为 CBOW。

🛠️ 常见问题排查

报错“未找到测试词汇”
原因：语料库太小，或者分词后词汇被 min_count 过滤掉了。
解决：确保 train.py 中 min_count=1。

相似词全是标点符号（如 ， 。）
原因：语料库太小，标点符号出现频率过高，导致模型认为标点符号和所有词都相似。
解决：本项目代码已内置 STOPWORDS 集合，会自动过滤常见标点。如果仍有生僻标点干扰，请在 src/train.py 的 STOPWORDS 集合中添加该符号。

找不到保存的模型文件
原因：运行脚本时的“当前工作目录”与项目根目录不一致。
解决：
    始终在项目根目录（即包含 src 和 data 文件夹的层级）运行命令。
    查看控制台输出的 [调试信息] 行，它会打印模型保存的绝对路径。

可视化图中文字乱码
原因：Matplotlib 默认字体不支持中文。
解决：在 src/visualize.py 开头添加以下代码（Windows 系统）：
    python
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

📝 许可证
本项目仅供学习和研究使用。
