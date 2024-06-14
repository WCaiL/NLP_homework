import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x.unsqueeze(1))
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(target.device)
        hidden, cell = self.encoder(source)

        input = target[:, 0] 
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input = target[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        
        return outputs




# class Seq2Seq(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout):
#         super(Seq2Seq, self).__init__()
#         self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
#         self.decoder = nn.LSTM(output_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         batch_size = src.shape[0]
#         trg_len = trg.shape[1]
#         trg_vocab_size = self.fc.out_features
        
#         outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
#         _, (hidden, cell) = self.encoder(src)
        
#         input = trg[:, 0]
        
#         for t in range(1, trg_len):
#             output, (hidden, cell) = self.decoder(input.unsqueeze(1), (hidden, cell))
#             output = self.fc(output.squeeze(1))
#             outputs[:, t] = output
#             teacher_force = random.random() < teacher_forcing_ratio
#             top1 = output.argmax(1)
#             input = trg[:, t] if teacher_force else top1
        
#         return outputs
