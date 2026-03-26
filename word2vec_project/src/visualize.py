import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec


# 设置中文字体（Windows 系统）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_word_vectors(model_path, output_path, target_words=None):
    # 加载模型
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        print("请先运行 train.py 训练模型。")
        return

    print("正在加载模型...")
    model = Word2Vec.load(model_path)

    # 如果用户没有指定词，尝试从模型中随机选取或选取常见字
    # 由于我们是按字训练的，这里选取一些常见的中文字
    if target_words is None:
        # 尝试选取一些可能在语料中的字
        potential_words = ['人工智能', '自然', '数据', '学习', '森林', '机器', '环境', '科技', '生命', '地球']
        target_words = [w for w in potential_words if w in model.wv]

        # 如果不够10个，就从词汇表中随机补全
        if len(target_words) < 10:
            all_words = list(model.wv.index_to_key)
            import random
            remaining = 10 - len(target_words)
            extra_words = random.sample([w for w in all_words if w not in target_words], min(remaining, len(all_words)))
            target_words.extend(extra_words)

    # 确保只取前10个
    target_words = target_words[:10]

    print(f"选定的 10 个词/字: {target_words}")

    # 获取向量
    vectors = []
    valid_words = []
    for word in target_words:
        if word in model.wv:
            vectors.append(model.wv[word])
            valid_words.append(word)
        else:
            print(f"警告: 词汇 '{word}' 不在模型中，已跳过。")

    if len(vectors) < 2:
        print("错误：有效的词向量太少，无法绘图。")
        return

    vectors = np.array(vectors)

    # 使用 t-SNE 降维到 2D
    print("正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(vectors) - 1))
    vectors_2d = tsne.fit_transform(vectors)

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='skyblue', s=100, alpha=0.7, edgecolors='b')

    # 添加标签
    for i, word in enumerate(valid_words):
        plt.annotate(word,
                     xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=12,
                     fontweight='bold')

    plt.title('Word2Vec 词向量分布图 (t-SNE 降维)', fontsize=16)
    plt.xlabel('维度 1')
    plt.ylabel('维度 2')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"图片已保存至: {output_path}")
    plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'word2vec.model')
    output_path = os.path.join(base_dir, 'results', 'word_vectors_plot.png')

    # 你可以手动指定想展示的10个字，如果语料足够大
    # my_words = ['人工智能', '深度学习', ...] # 如果用了jieba分词
    plot_word_vectors(model_path, output_path)