import torch
from train_LSTM import load_corpus, tokenize_corpus, build_vocab, LSTMModel, prepare_data
import pickle
import jieba


if __name__ == '__main__':
    
    with open('tokenized_corpus.pkl', 'rb') as f:
        tokenized_corpus = pickle.load(f)

    print([len(doc) for doc in tokenized_corpus])
    
    vocab, word2idx, idx2word = build_vocab(tokenized_corpus)
    sequences = prepare_data(tokenized_corpus, word2idx, seq_length=20)
    print(len(sequences))


    stop = [line.strip() for line in open('cn_stopwords.txt', encoding="utf-8").readlines()]
    stop.append(' ')
    stop.append('\n')
    stop.append('\u3000')
    
    # embed_size = 128
    # hidden_size = 256
    # num_layers = 2
    # vocab_size = len(vocab)

    # model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
    # model.load_state_dict(torch.load('lstm_epoch58.pth'))


    # text = '杨过'
    # words = list(jieba.cut(text, cut_all=False))
    # words = [word for word in words if word not in stop]
    # seq = [word2idx[word] for word in words]
    # seq = torch.tensor(seq).unsqueeze(0)
    # # 获取概率前十
    # output = model(seq)
    # _, topk = torch.topk(output, 10)
    # topk = topk.squeeze().tolist()
    # similar_words = [(idx2word[i], output[0][i]) for i in topk]
    # print(similar_words)

    
