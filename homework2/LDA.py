import gensim
from gensim import corpora
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import jieba
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

# 读取数据集
def load_data(Ktoken, mode):
    inf = open("./datasets_cn/inf.txt", "r", encoding="gb18030").read()  # gb18030 utf-8
    inf = inf.split(',')
    label_map = {}
    for i, name in enumerate(inf):
        label_map[name] = i
    # print(label_map)
    stop = [line.strip() for line in open('cn_stopwords.txt', encoding="utf-8").readlines()]
    stop.append(' ')
    stop.append('\n')
    stop.append('\u3000')
    documents = []
    labels = []
    all_data = []
    all_label = []
    sum = 0
    for name in tqdm(inf):
        with open("./datasets_cn/" + name + ".txt", "r", encoding="gb18030") as f:
            txt = f.read()
            ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'
            txt = txt.replace(ad, '') 
            sum += len(txt)
            if mode == 'word':
                words_all = jieba.lcut(txt)
                words_list = []
                i = 0
                for word in words_all:
                    if word not in stop:
                        words_list.append(word)
                        i += 1
                        if i == Ktoken:
                            documents.append(words_list)
                            labels.append(label_map[name])
                            words_list = []
                            i = 0
            if mode == 'char':
                chars_list = []
                i = 0
                for char in txt:
                    if char not in stop:
                        chars_list.append(char)
                        i += 1
                        if i == Ktoken:
                            documents.append(chars_list)
                            labels.append(label_map[name])
                            chars_list = []
                            i = 0
    random_way = random.sample(range(len(documents)), 1000)
    for i in range(len(documents)):
        if i in random_way:
            all_data.append(documents[i])
            all_label.append(labels[i])
    return all_data, all_label


def train_lda_model(documents, num_topics):

    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    print('Training LDA model')
    # lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=20, workers=4)
    print('Finished training LDA model')

    return lda_model


def get_document_topic_distribution(lda_model, documents):
    topic_distributions = []
    for doc in documents:
        bow_vector = lda_model.id2word.doc2bow(doc)
        topic_distribution = lda_model.get_document_topics(bow_vector, minimum_probability=0.0)
        topic_distribution = [score for _, score in topic_distribution]
        topic_distributions.append(topic_distribution)
    return np.array(topic_distributions)


def evaluate_classification_performance(X, y, classifier):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X, y, cv=kf)
    return scores.mean()

if __name__ == "__main__":
    K_values = [20, 100, 500, 1000, 2500]
    num_topics_values = [8, 16, 32, 64]
    mode_list = ['word', 'char']
    acc_list_KW = {20: [], 100: [], 500: [], 1000: [], 2500: []}
    acc_list_TW = {8: [], 16: [], 32: [], 64: []}
    acc_list_KC = {20: [], 100: [], 500: [], 1000: [], 2500: []}
    acc_list_TC = {8: [], 16: [], 32: [], 64: []}
    all_data_w, all_label_w = load_data(2500, 'word')
    all_data_c, all_label_c = load_data(2500, 'char')
    for K in K_values:
        for num_topics in num_topics_values:
            for mode in mode_list:
                print(f"K = {K}, T = {num_topics}, Mode = {mode}")
                # all_data, all_label = load_data(K, mode)
                if mode == 'word':
                    all_data, all_label = [row[:K] for row in all_data_w], all_label_w
                elif mode == 'char':
                    all_data, all_label = [row[:K] for row in all_data_c], all_label_c
                lda_model = train_lda_model(all_data, num_topics)
                all_x_lda = get_document_topic_distribution(lda_model, all_data)
                accuracy = evaluate_classification_performance(all_x_lda, all_label, MultinomialNB())
                print("Accuracy:", accuracy)
                print("=" * 50)
                with open("result.txt", "a") as f:
                    f.write(f"K = {K}, T = {num_topics}, Mode = {mode}\n")
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write("=" * 50 + "\n")
                if mode == 'word':
                    acc_list_KW[K].append(accuracy)
                    acc_list_TW[num_topics].append(accuracy)
                if mode == 'char':
                    acc_list_KC[K].append(accuracy)
                    acc_list_TC[num_topics].append(accuracy)

    # 画出相同K值下不同T值的准确率
    plt.figure()
    for K in K_values:
        plt.plot(num_topics_values, acc_list_KW[K], label=f"word K = {K}")
        plt.plot(num_topics_values, acc_list_KC[K], label=f"char K = {K}", linestyle='-.')
    plt.legend()
    plt.xlabel("Number of topics")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of topics (Word)")
    plt.savefig("Accuracy vs Number of topics (Word).png")
    # 画出相同T值下不同K值的准确率
    plt.figure()
    for num_topics in num_topics_values:
        plt.plot(K_values, acc_list_TW[num_topics], label=f"word T = {num_topics}")
        plt.plot(K_values, acc_list_TC[num_topics], label=f"char T = {num_topics}", linestyle='-.')
    plt.legend()
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs K (Word)")
    plt.savefig("Accuracy vs K (Word).png")



