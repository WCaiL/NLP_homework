import torch
from torch.utils.data import DataLoader
from seq2seq_model import Seq2Seq, Encoder, Decoder
from train_seq2seq import TextDataset, load_corpus
import random


def generate_text(model, start_string, gen_length, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    generate_text = start_string
    while True:
        if len(generate_text) > gen_length:
            break
        generate_text_index = [dataset.char2idx[s] for s in generate_text]
        output = model(torch.tensor(generate_text_index[:-1]).unsqueeze(0).to(device), torch.tensor(generate_text_index[1:]).unsqueeze(0).to(device), 0)
        top1 = output.argmax(2)[:, -1].item()
        next_char = dataset.idx2char[top1]
        generate_text += next_char

    
    return generate_text

# def generate_text(model, start_string, gen_length, dataset):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.eval()
#     input_eval = torch.tensor([dataset.char2idx[s] for s in start_string]).unsqueeze(1).to(device)
#     print(input_eval.shape)
#     hidden, cell = model.encoder(input_eval)
#     input = input_eval[:, -1]
#     print(input.shape, hidden.shape, cell.shape)

#     generated_text = start_string

#     for _ in range(gen_length):
#         output, hidden, cell = model.decoder(input, hidden, cell)
#         top = output.argmax(1)
#         top1 = random.choice(top)
#         next_char = dataset.idx2char[top1.item()]
#         generated_text += next_char
#         input = top
    
#     return generated_text

if __name__ == "__main__":
    text = load_corpus()
    seq_length = 30
    dataset = TextDataset(text, seq_length)
    # dataset = torch.load('dataset10.pth')
    start_string = "郭靖在襄阳城上"
    gen_length = 500
    
    vocab_size = len(dataset.vocab)
    embed_size = 128
    hidden_size = 256
    num_layers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
    decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(torch.load('./model/seq2seq_best.pth')['model_state_dict'])

    generated_text = generate_text(model, start_string, gen_length, dataset)
    print(generated_text)



