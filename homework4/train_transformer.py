import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformer_model import TransformerModel
import pickle

class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.vocab = sorted(set(text))
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        self.text_as_int = [self.char2idx[c] for c in text]
    
    def __len__(self):
        return len(self.text_as_int) - self.seq_length
    
    def __getitem__(self, idx):
        return (torch.tensor(self.text_as_int[idx:idx+self.seq_length]),
                torch.tensor(self.text_as_int[idx+1:idx+self.seq_length+1]))

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
            txt = txt.replace(' ', '')
            txt = txt.replace('\n', '')
            txt = txt.replace('□', '')
            corpus += txt
    return corpus


if __name__ == '__main__':
    text = load_corpus()
    seq_length = 30
    dataset = TextDataset(text, seq_length)
    torch.save(dataset, 'dataset.pth')
    # dataset = torch.load('dataset10.pth')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(dataset.vocab)
    print("vocab_size: ", vocab_size)
    embed_size = 128
    hidden_size = 256
    num_layers = 2
    num_heads = 8

    model = TransformerModel(vocab_size, embed_size, num_heads, num_layers, hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # model_load = torch.load('transformer_8.pth')
    # model.load_state_dict(model_load['model_state_dict'])
    # optimizer.load_state_dict(model_load['optimizer_state_dict'])

    start = 0
    epochs = 50
    loss_list = []
    model.train()
    for epoch in range(start, epochs):
        loss_sum = 0
        for source, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            source, target = source.to(device), target.to(device)
            output = model(source, target)
            output = output.reshape(-1, output.shape[2])
            target = target.reshape(-1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            loss_sum += loss.item()
        loss_avg = loss_sum / len(dataloader)
        pickle.dump(loss_list, open('loss_list.pkl', 'wb'))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_avg:.4f}")
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'transformer_{}.pth'.format(epoch+1))
