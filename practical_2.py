# %%
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import os
import urllib
import zipfile
import lxml.etree
import re
from collections import Counter

# %%
if not os.path.isfile('ted_en-20160408.xml'):
    urllib.request.urlretrieve("https://github.com/oxford-cs-deepnlp-2017/practical-1/blob/master/ted_en-20160408.xml?raw=true", filename="ted_en-20160408.xml")

# %%
doc = lxml.etree.parse('ted_en-20160408.xml')
input_text = doc.xpath('//content/text()')
label = doc.xpath('//head/keywords/text()')
del doc
len(input_text)

# %%
# Preprocess sentences to exclude all characters except alphabets and numbers
texts = [re.sub(r'\([^)]*\)', '',text) for text in input_text]
texts = [re.sub('r([^a-zA-Z0-9\s])',' ',text) for text in texts] #Included '.'
texts = [re.sub('[^a-zA-Z0-9\']',' ',text) for text in texts] #To replace '.' with ' '
texts = [re.sub('[^a-zA-Z0-9 ]','',text) for text in texts]
texts = [text.lower() for text in texts] #uppercase->lowercase

# %%
texts[2069][:160]

# %%
text_labels = zip(texts,label)
texts = [text_label for text_label in text_labels if len(text_label[0]) > 500]
print('number of text greater than 500 words are:',len(texts))

# %%
texts,labels = zip(*texts)

# %%
texts[0]

# %%
words = [words for text in texts for words in text.split()]
words_count = Counter(words)
words_most_common =[word for word,count in words_count.most_common(100)]
words_least_common = [word for word,count in words_count.most_common() if count==1]

# %%
to_remove = words_most_common + words_least_common
words_to_remove = set(to_remove)
tokens = [word for word in words if word not in words_to_remove] #will be used during T-SNE
print('size of Token:',len(tokens)) 

# %%
texts = [[word for word in text.split() if word not in words_to_remove]for text in texts]

# %%
# Encode labels as ['ooo', 'Too', 'oEo', 'ooD', 'TEo', 'ToD', 'oED', 'TED']
label_coded = ['ooo']*len(labels)
for i,keyword in enumerate(labels):
    key = keyword.split(', ')
    label = list(label_coded[i])
    if 'technology' in key:
        label[0] = 'T'
    if 'entertainment' in key:
        label[1] = 'E'
    if 'design' in key:
        label[2] = 'D'
    else:
        pass
    label_coded[i] =''.join(label) 

# %%
count_labels=Counter(label_coded)
label_count = [word_count for word_count in count_labels.most_common()]
label_count

# %%
one_hotted = np.zeros(shape=(len(labels),8),dtype='int16')
label_lookup = ['ooo', 'Too', 'oEo', 'ooD', 'TEo', 'ToD', 'oED', 'TED']

# %%
label_lookup = ['ooo', 'Too', 'oEo', 'ooD', 'TEo', 'ToD', 'oED', 'TED']
for i,label in enumerate(label_coded):
    one_hotted[i][label_lookup.index(label)] = 1
print(one_hotted[:10])    

# %%
tokens.append('<UNK>')
tokens.append('<PAD>')

# %%
vocab = list(set(tokens))

# %%
print('size of vocabulary:',len(vocab))
id2word = dict(enumerate(vocab))
word2id = dict((val,key) for (key,val) in id2word.items())

# %%
# Stripping Text to fall within length of 500; incase if it is shorter then padd with '<UNK>'
length = 500 #sentence length
stripped_text = []#np.zeros((len(texts),length)
for i,text in enumerate(texts):
    inputs = []
    if len(text) >= 500:
        inputs.extend(text[:500])
    else:
        extra_length = 500-len(text)
        extra = ['<PAD>']*extra_length
        word_with_extra = text + extra
        inputs.extend(word_with_extra)
    stripped_text.append(inputs) 

# %%
stripped_length = len(stripped_text)
print(stripped_length)

# %%
for i,code in enumerate(label_coded):
    one_hotted[i][label_lookup.index(code)] = 1

# %%
inputs = []
text_ids = []
for text in stripped_text:
    for word in text:
        i = word2id[word]
        inputs.append(i)
    text_ids.append(inputs)
    inputs = []

# %%
text_ids[0][100] , id2word[text_ids[0][100]], stripped_text[0][100], word2id['<UNK>']

# %%
data = list(zip(text_ids,one_hotted))
tr_size = round(0.8*len(data))
vl_size = round(0.1*len(data))
te_size = tr_size + vl_size
n_classes = one_hotted.shape[1]
train_Xy , val_Xy , test_Xy = [],[],[]
for i in np.arange(n_classes):
    j = np.zeros(n_classes)
    j[i] = 1
    temp = [text_ohe for text_ohe in data if text_ohe[1][i]==j[i]]
    temp_len = len(temp)
    tr_split = round(temp_len*0.8)
    val_split = round(temp_len*0.9)
    train_Xy.extend(temp[:tr_split])
    val_Xy.extend(temp[tr_split:val_split])
    test_Xy.extend(temp[val_split:])
random.shuffle(train_Xy)
random.shuffle(val_Xy)
random.shuffle(test_Xy)

# %%
train_ids = torch.from_numpy(np.array([i[0] for i in train_Xy]))
train_labels = [i[1] for i in train_Xy]
val_ids = torch.from_numpy(np.array([i[0] for i in val_Xy]))
val_labels = [i[1] for i in val_Xy]
test_ids = torch.from_numpy(np.array([i[0] for i in test_Xy]))
test_labels = [i[1] for i in test_Xy]
train_labels = torch.tensor([label.tolist().index(1) for label in train_labels], dtype=torch.long)
val_labels = torch.tensor([label.tolist().index(1) for label in val_labels], dtype=torch.long)
test_labels = torch.tensor([label.tolist().index(1) for label in test_labels], dtype=torch.long)

# %%
len(val_labels)

# %%
class TED_Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = ID
        y = self.labels[index]

        return X, y

# %%
train_Xy = TED_Dataset(train_ids,train_labels)
val_Xy = TED_Dataset(val_ids,val_labels)
test_Xy = TED_Dataset(test_ids,test_labels)

# %%
train_dataloader = DataLoader(train_Xy, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_Xy, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_Xy, batch_size=64, shuffle=False)

# %%
len(train_Xy)

# %%
len(val_Xy)

# %%
len(test_Xy)

# %%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %%
class TEDModel(nn.Module):
    def __init__(self,vocab_size,embedding_dimension=320):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dimension)
        self.hidden_layer= nn.Sequential(
            nn.Linear(320,100,bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.4),
            nn.Linear(100, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.4),
            nn.Linear(64,8,bias=True)
        )


    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        out = self.hidden_layer(x)
        predicted_class = torch.argmax(out, dim=1)
        return out



# %%
ted_model = TEDModel(37328)
num_epochs = 10
batch_size=64
loss_fn = nn.CrossEntropyLoss()

learning_rate=1e-3
optimizer = optim.Adam(ted_model.parameters(), lr=learning_rate)

# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), (batch * batch_size + len(X))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %%
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, ted_model, loss_fn, optimizer)
    test_loop(test_dataloader, ted_model, loss_fn)
print("Done!")

# %%
torch.save(ted_model.state_dict(), "model2.pt")

# %%
model = TEDModel(37328)
model.load_state_dict(torch.load("model2.pt", weights_only=True))
model.eval()

# %%



