import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, embed_size, seq_length, num_heads, num_layers, ff_hidden_size, dropout=0.1):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.positional_encoding = self.create_positional_encoding(embed_size, seq_length)
#         self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, ff_hidden_size, dropout)
#         self.fc = nn.Linear(embed_size, vocab_size)

#     def create_positional_encoding(self, embed_size, max_len):
#         pe = torch.zeros(max_len, embed_size)
#         for pos in range(max_len):
#             for i in range(0, embed_size, 2):
#                 pos_t = torch.tensor(pos, dtype=torch.float32)
#                 pe[pos, i] = torch.sin(pos_t / (10000 ** (i / embed_size)))
#                 if i + 1 < embed_size:
#                     pe[pos, i + 1] = torch.cos(pos_t / (10000 ** ((i + 1) / embed_size)))
#         return pe.unsqueeze(0)


#     def forward(self, src, tgt):
#         src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :].to(src.device)
#         tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
#         src = src.permute(1, 0, 2)  # (seq_length, batch_size, embed_size)
#         tgt = tgt.permute(1, 0, 2)  # (seq_length, batch_size, embed_size)
#         output = self.transformer(src, tgt)
#         output = output.permute(1, 0, 2)  # (batch_size, seq_length, embed_size)
#         output = self.fc(output)
#         return output
    

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden_size, max_len=5000, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = self.create_positional_encoding(embed_size, max_len)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, ff_hidden_size, dropout)
        self.fc = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size

    def create_positional_encoding(self, embed_size, max_len):
        pe = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pos_t = torch.tensor(pos, dtype=torch.float32)
                pe[pos, i] = torch.sin(pos_t / (10000 ** (i / embed_size)))
                if i + 1 < embed_size:
                    pe[pos, i + 1] = torch.cos(pos_t / (10000 ** ((i + 1) / embed_size)))
        return pe.unsqueeze(0)

    def forward(self, src, tgt):
        src = self.embedding(src) * math.sqrt(self.embed_size) + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt = self.embedding(tgt) * math.sqrt(self.embed_size) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        src = src.permute(1, 0, 2)  # (seq_length, batch_size, embed_size)
        tgt = tgt.permute(1, 0, 2)  # (seq_length, batch_size, embed_size)
        output = self.transformer(src, tgt)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, embed_size)
        output = self.fc(output)
        return output
