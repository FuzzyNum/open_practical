import glob
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import string

# Load the corpus
corpus = ""
for filename in glob.glob('TED_transcripts/*.txt'):
    with open(filename, 'r') as f:
        corpus += 2 * '\n' + f.read().replace('\n', ' ')

allowed_characters = string.ascii_lowercase + string.ascii_uppercase + " "
corpus = ''.join([ch for ch in corpus if ch in allowed_characters])

# Training data split
train_split = 0.98
train_size = int(train_split * len(corpus))

train_data = corpus[:train_size]
val_data = corpus[train_size:]

# Vocabulary and other parameters
vocab = list(set(corpus))
input_dim = len(vocab)
hidden_dim = 256
sample_size = 256
batch_size = 64
num_layers = 2
num_epochs = 100
learning_rate = 1e-3
char2idx = {ch: idx for idx, ch in enumerate(vocab)}
idx2char = {idx: ch for ch, idx in char2idx.items()}


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)  # (max_len, 1, d_model)

    def forward(self, x):
        return x + self.pe[:x.size(0)].to(x.device)


# Transformer-based network
class network(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(hidden_dim, input_dim)

    def reset(self):
        pass  # no hidden states for transformer

    def forward(self, ip):
        # ip: (batch, seq_len)
        ip = ip.transpose(0, 1)  # -> (seq_len, batch)
        emb = self.embedding(ip) * np.sqrt(hidden_dim)
        emb = self.pos_encoder(emb)  # (seq_len, batch, hidden)
        out = self.transformer(emb)  # (seq_len, batch, hidden)
        out = self.output(out)  # (seq_len, batch, vocab)
        return out.view(-1, input_dim)  # flatten to (seq_len * batch, vocab)


# Generate a random sample from the dataset
def random_sample(data):
    start = np.random.randint(0, len(data) - sample_size - 1)
    sample = data[start:start + sample_size]
    label = data[start + 1:start + sample_size + 1]
    return sample, label


# Generate a batch of training data
def genbatch(dataset):
    ip = torch.zeros(batch_size, sample_size).long()
    target = torch.zeros(batch_size, sample_size).long()
    for b in range(batch_size):
        ip_sample, target_sample = random_sample(dataset)
        for t in range(sample_size):
            ip[b, t] = char2idx[ip_sample[t]]
            target[b, t] = char2idx[target_sample[t]]
    return ip, target


# Validation function
def validate(model):
    val_loss = 0
    val_batch_num = 20
    model.eval()
    with torch.no_grad():
        for _ in range(val_batch_num):
            model.reset()
            val_ip, val_target = genbatch(val_data)
            val_target = val_target.view(-1)
            val_pred = model(val_ip)
            val_loss += criterion(val_pred, val_target).item()
    model.train()
    return val_loss / val_batch_num


# Save checkpoint function
def save_checkpoint(model, val_loss, e, optimizer):
    cp_name = f'e{e+1}_{val_loss:.4f}.pth'
    cp_path = cp_dir + cp_name
    opt_path = opt_dir + cp_name
    torch.save(model.state_dict(), cp_path)
    torch.save(optimizer.state_dict(), opt_path)


# Main script
if __name__ == "__main__":
    TED = network(input_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(TED.parameters(), lr=learning_rate)

    cp_dir = "./checkpoints/"
    os.makedirs(cp_dir, exist_ok=True)

    opt_dir = "./opt/"
    os.makedirs(opt_dir, exist_ok=True)

    batches_per_epoch = 200

    for e in range(num_epochs):
        print(f"\nEpoch {e+1}/{num_epochs}")
        batch_losses = []

        for _ in tqdm(range(batches_per_epoch), desc="Training", leave=False):
            TED.reset()
            ip, target = genbatch(train_data)
            target = target.view(-1)
            pred = TED(ip)

            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        avg_train_loss = sum(batch_losses) / len(batch_losses)
        val_loss = validate(TED)

        print(f"  Avg training loss: {avg_train_loss:.4f}")
        print(f"  Validation loss:   {val_loss:.4f}")

        save_checkpoint(TED, val_loss, e, optimizer)
