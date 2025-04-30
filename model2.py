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

allowed_characters = string.ascii_lowercase + string.ascii_uppercase + " "  # only letters and space
corpus = ''.join([ch for ch in corpus if ch in allowed_characters])
# Training data split
train_split = 0.98  # fraction of training data
train_size = int(train_split * len(corpus))

print(len(set(corpus)))
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
onehot_mat = torch.eye(input_dim)  # matrix for one-hot lookup
char2idx = {ch: idx for idx, ch in enumerate(vocab)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

# Network definition
class network(nn.Module):
    def __init__(self, input_dim, hi_dim, num_layers):
        super(network, self).__init__()
        self.ip_dim = input_dim
        self.hi_dim = hi_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.ip_dim, self.hi_dim)
        self.lstm = nn.LSTM(self.hi_dim, self.hi_dim, self.num_layers, batch_first=True)
        self.linear = nn.Linear(hi_dim, input_dim)

    def reset(self):
        self.h0 = torch.zeros(self.num_layers, self.batch_size, self.hi_dim)
        self.c0 = torch.zeros(self.num_layers, self.batch_size, self.hi_dim)
        self.hidden = self.h0, self.c0

    def forward(self, ip):
        emb = self.embedding(ip)
        op_pred, self.hidden = self.lstm(emb, self.hidden)
        op_pred = self.linear(op_pred).view(-1, input_dim)
        return op_pred

# Generate a random sample from the dataset
def random_sample(data):
    start = np.random.randint(0, len(data) - sample_size)
    end = start + sample_size
    sample = data[start:end]
    label = data[start+1:end+1]
    return sample, label

# Generate a batch of training data
def genbatch(dataset):
    ip = torch.zeros(batch_size, sample_size).long()      # (batch, seq_len)
    target = torch.zeros(batch_size, sample_size).long()  # (batch, seq_len)
    for b in range(batch_size):
        ip_sample, target_sample = random_sample(dataset)
        for letter in range(sample_size):
            ip[b, letter] = char2idx[ip_sample[letter]]
            target[b, letter] = char2idx[target_sample[letter]]
    return ip, target


# Validation function
def validate(model):
    val_copy = val_data
    val_loss = 0
    val_batch_num = 20  # number of mini-batches of validation set to evaluate loss
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients required for validation
        for i in range(val_batch_num):
            model.reset()
            val_ip, val_target = genbatch(val_copy)
            val_ip = val_ip  # Tensor already, no need for Variable
            val_target = val_target.view(-1)
            val_pred = model(val_ip)
            val_loss += criterion(val_pred, val_target)
    val_loss = val_loss.item() / val_batch_num  # Convert tensor to Python float
    model.train()  # Set the model back to training mode
    return val_loss

# Save checkpoint function
def save_checkpoint(model, val_loss, e, optimizer):
    cp_name = 'e{}_{:.4f}.pth'.format(e+1, val_loss)
    cp_path = cp_dir + cp_name
    opt_path = opt_dir + cp_name
    torch.save(model.state_dict(), cp_path)  # Save model
    torch.save(optimizer.state_dict(), opt_path)  # Save optimizer

# Main script
if __name__ == "__main__":
    TED = network(input_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(TED.parameters(), lr=learning_rate)

    cp_dir = "./checkpoints/"
    if not os.path.isdir(cp_dir):
        os.mkdir(cp_dir)

    opt_dir = "./opt/"
    if not os.path.isdir(opt_dir):
        os.mkdir(opt_dir)

    batches_per_epoch = 500

    for e in range(num_epochs):
        print(f"\nEpoch {e+1}/{num_epochs}")
        batch_losses = []

        for batch in tqdm(range(batches_per_epoch), desc="Training", leave=False):
            TED.reset()
            ip, target = genbatch(train_data)
            ip = ip  # Tensor already
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