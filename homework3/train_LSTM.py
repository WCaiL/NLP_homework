import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import jieba
from collections import Counter
import pickle


def load_corpus():
    print('begin load corpus')
    inf = open("./datasets_cn/inf.txt", "r", encoding="gb18030").read()  # gb18030 utf-8
    inf = inf.split(',')
    corpus = []
    for name in tqdm(inf):
        with open("./datasets_cn/" + name + ".txt", "r", encoding="gb18030") as f:
            txt = f.read()
            ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'
            txt = txt.replace(ad, '')
            corpus.append(txt)
    return corpus

def tokenize_corpus(corpus):
    print('begin tokenize corpus')
    tokenized_corpus = []
    stop = [line.strip() for line in open('cn_stopwords.txt', encoding="utf-8").readlines()]
    stop.append(' ')
    stop.append('\n')
    stop.append('\u3000')
    for document in tqdm(corpus):
        tokens = jieba.lcut(document, cut_all=False)
        tokens = [token for token in tokens if token not in stop]
        tokenized_corpus.append(tokens)
    with open('tokenized_corpus.pkl', 'wb') as f:
        pickle.dump(tokenized_corpus, f)
    return tokenized_corpus


def build_vocab(tokenized_corpus, min_freq=5):
    counter = Counter([token for document in tokenized_corpus for token in document])
    vocab = [token for token, freq in counter.items() if freq >= min_freq]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return vocab, word2idx, idx2word

def prepare_data(tokenized_corpus, word2idx, seq_length=50):
    sequences = []
    for document in tokenized_corpus:
        encoded = [word2idx[word] for word in document if word in word2idx]
        for i in range(seq_length, len(encoded)):
            seq = encoded[i-seq_length:i]
            target = encoded[i]
            sequences.append((seq, target))
    return sequences



class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq), torch.tensor(target)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x




if __name__ == '__main__':
    # corpus = load_corpus()
    # tokenized_corpus = tokenize_corpus(corpus)

    with open('tokenized_corpus.pkl', 'rb') as f:
        tokenized_corpus = pickle.load(f)
    # print(len(tokenized_corpus), len(tokenized_corpus[0]))
    # print(tokenized_corpus[0][:10])
    vocab, word2idx, idx2word = build_vocab(tokenized_corpus)
    # print(vocab[:10])
    # print(list(word2idx.items())[:10])
    # print(list(idx2word.items())[:10])
    # print(len(vocab))
    # print(len(word2idx))
    # print(len(idx2word))
    # print(tokenized_corpus[0][:10])

    sequences = prepare_data(tokenized_corpus, word2idx, seq_length=10)
    dataset = TextDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    embed_size = 128        # 200
    hidden_size = 256
    num_layers = 2
    vocab_size = len(vocab)

    model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
    model.load_state_dict(torch.load('lstm_epoch60.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_epochs = 61
    num_epochs = 100
    loss_list = []

    for epoch in range(start_epochs, num_epochs):
        loss_sum = 0
        for seq, target in tqdm(dataloader):
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            loss_avg = loss_sum / len(dataloader)
            loss_list.append(loss.item())
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, avg_Loss: {loss_avg}')

        with open('loss_list.pkl', 'wb') as f:
            pickle.dump(loss_list, f)

        torch.save(model.state_dict(), 'lstm_epoch{}.pth'.format(epoch+1))
