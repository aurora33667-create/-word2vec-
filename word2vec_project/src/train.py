import os
import logging
import jieba
from gensim.models import Word2Vec

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 【新增】定义需要过滤的标点符号和停用词
# 你可以根据需要往这里面添加不想要的字符
STOPWORDS = set(['，', '。', '！', '？', '；', '：', '、', '“', '”', '（', '）', '(', ')', ' ', '\n', '\t'])


def load_data(file_path):
    """加载并预处理数据 - 去除标点版"""
    sentences = []

    if not os.path.exists(file_path):
        print(f"⚠️ 文件未找到: {file_path}，正在自动创建测试数据...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("人工智能正在改变世界，深度学习是核心技术。\n")
            f.write("自然语言处理让计算机读懂人类语言。\n")
            f.write("保护自然环境非常重要，我们要爱护森林。\n")
        print("✅ 测试数据创建成功。")

    print(f"📂 正在读取数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"📄 读取到总行数: {len(lines)}")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 1. Jieba 分词
        words = list(jieba.lcut(line))

        # 2. 【核心修改】过滤掉标点符号和停用词
        # 只保留那些 不在 STOPWORDS 集合中 且 长度大于0 的词
        clean_words = [w for w in words if w not in STOPWORDS and w.strip()]

        if len(clean_words) > 0:
            sentences.append(clean_words)

    print(f"✅ 数据预处理完成，有效句子数量: {len(sentences)}")

    # 打印前3句看看效果，确认标点已消失
    if len(sentences) > 0:
        print(f"🔍 清洗后样例: {sentences[:3]}")

    return sentences


def train_model(sentences, vector_size=100, window=2, min_count=1, epochs=50):
    """
    训练 Word2Vec 模型
    注意：对于小语料，调大 epochs (迭代次数) 有助于模型收敛
    """
    if len(sentences) == 0:
        print("❌ 错误：句子列表为空，无法训练！")
        return None

    print("🚀 开始训练 Word2Vec 模型...")
    try:
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,  # 小语料窗口设小一点，捕捉紧密关系
            min_count=min_count,
            workers=4,
            sg=1,  # Skip-gram
            epochs=epochs  # 增加迭代次数
        )
        print("✅ 训练完成！")
        return model
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        return None


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'corpus.txt')
    model_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(model_dir, 'word2vec.model')

    os.makedirs(model_dir, exist_ok=True)

    # 1. 加载数据（会自动过滤标点）
    sentences = load_data(data_path)

    # 2. 训练
    model = train_model(sentences)

    if model is None: return

    # 3. 保存
    model.save(model_path)
    print(f"💾 模型已保存至: {model_path}")

    # 4. 测试
    test_words = ['人工智能', '自然', '数据', '学习', '森林', '科技']
    found_word = None

    for w in test_words:
        if w in model.wv:
            found_word = w
            break

    if found_word:
        try:
            # 查找最相似的词
            similar_words = model.wv.most_similar(found_word, topn=5)
            print(f"\n🎉 最终测试结果！与 '{found_word}' 最相似的词:")
            for word, score in similar_words:
                print(f"   - {word}: {score:.4f}")
        except Exception as e:
            print(f"\n⚠️ 查找出错: {e}")
    else:
        print(f"\n❌ 测试词汇都不在模型中。")


if __name__ == "__main__":
    main()