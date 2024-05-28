import torch

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from train_LSTM import load_corpus, tokenize_corpus, build_vocab, LSTMModel

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
font_name = "simhei"
matplotlib.rcParams['font.family']= font_name
matplotlib.rcParams['axes.unicode_minus']=False
import pickle

def find_similar_words(word, word2idx, idx2word, embeddings, top_n=10):
    idx = word2idx[word]
    word_vec = embeddings[idx].unsqueeze(0)
    similarities = cosine_similarity(word_vec, embeddings)[0]
    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
    similar_words = [(idx2word[i], similarities[i]) for i in similar_indices]
    return similar_words

def plot_embedding(embeddings, word2idx, words, title='Word Embeddings'):
    idx = []
    for word in words:
        if word in word2idx:
            idx.append(word2idx[word])
    selected_embeddings = embeddings[idx]

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    plt.figure(figsize=(14, 10))
    for i, word in enumerate(words):
        x, y = reduced_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')

    plt.title(title)
    plt.show()

def plot_embeddings(embeddings, idx2word, top_n=500):
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings[:top_n])
    
    plt.figure(figsize=(10, 10))
    for i in range(top_n):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(idx2word[i], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    plt.show()




if __name__ == '__main__':

    # corpus = load_corpus()
    # tokenized_corpus = tokenize_corpus(corpus)
    with open('tokenized_corpus.pkl', 'rb') as f:
        tokenized_corpus = pickle.load(f)

    vocab, word2idx, idx2word = build_vocab(tokenized_corpus)

    embed_size = 128   #128
    hidden_size = 256
    num_layers = 2     # 2
    vocab_size = len(vocab)

    model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
    
    model.load_state_dict(torch.load('model/best.pth'))
    word_embeddings = model.embedding.weight.data

    similar_words = find_similar_words('韦小宝', word2idx, idx2word, word_embeddings)
    print(similar_words)

    word = ['张无忌', '乔峰', '郭靖', '杨过', '令狐冲', '韦小宝']
    # plot_embedding(word_embeddings, idx2word, word)
    # plot_embeddings(word_embeddings, idx2word, top_n=500)

