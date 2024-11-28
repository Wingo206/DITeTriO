import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import random

C, H, W = 1, 20, 10

channel_1 = 32
channel_2 = 16
channel_3 = 8
kernel_size_1 = 9
pad_size_1 = 4
kernel_size_2 = 7
pad_size_2 = 3
kernel_size_3 = 5
pad_size_3 = 2

hidden_layer_1 = 100
hidden_layer_2 = 100
output = 8

leakySlope = 1e-2
lr = 1e-3
epochs = 20
bs = 64

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

class board_model(nn.Module):
    def __init__(self):
        super(board_model,self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(C, channel_1, (kernel_size_1, kernel_size_1), padding = pad_size_1),
            nn.LeakyReLU(negative_slope=leakySlope),
            nn.Conv2d(channel_1, channel_2, (kernel_size_2, kernel_size_2), padding = pad_size_2),
            nn.LeakyReLU(negative_slope=leakySlope),
            nn.Conv2d(channel_2, channel_3, (kernel_size_3, kernel_size_3), padding = pad_size_3),
            nn.LeakyReLU(negative_slope=leakySlope),
            nn.Flatten()
        )
    
    def foward(self, x):
        return self.seq1(x)

class prob_model(nn.Module):
    def __init__(self):
        super(prob_model,self).__init__()

        self.seq1 = nn.Sequential(
            nn.Linear(channel_3*H*W+21, hidden_layer_1),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.Linear(hidden_layer_2, output)
        )

    def foward(self, x, new_data):
        x = torch.cat(x, new_data)
        return self.seq1(x)
    
class combined_model(nn.Module):
    def __init__(self):
        super(combined_model,self).__init__()

        self.board_model = board_model
        self.prob_model = prob_model

    def foward(self, x, new_data):
        x = self.board_model(x)
        return self.prob_model(x, new_data)

def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight)
        nn.init.zeros_(model.bias)

optimizer = optim.Adam(prob_model.parameters(), lr)
model = combined_model.to(device)
model.apply(init_weights)
loss_fn = nn.MSELoss()

#need to write a dataloader of some sort

class TetrDataset(Dataset):
    def __init__(self, df):
        self.board = df[:,0:199]
        self.other = df[:,200:220]
        self.label = df[:,221:228]
    
    def __len__(self):
        return len(self.labels)
    
    def __get_item__(self, idx):
        board = self.board.iloc[idx]
        other = self.other.iloc[idx]
        if idx != len(self.labels)-1:
            label = self.label.iloc[idx+1]
        else:
            label = self.label.iloc[0]

        return board, other, label

gulagland = 30
data = pd.read_csv() #input csv here
data = data[:-gulagland]
dataset = TetrDataset(data)

random.seed(1)
shuffled = random.shuffle(dataset)
train_data = shuffled[0:0.9*len(shuffled)]
val_data = shuffled[0.9*len(shuffled):]
train_dataset = DataLoader(train_data, batchsize = bs,shuffle = True)
val_dataset = DataLoader(val_data, batchsize = bs, shuffle = True)


def train(device, model, optimizer, loss_fn, dataloader):
    model.train()
    total_loss = 0
    for batch_idx, (board, other, label) in enumerate(dataloader):
        board, other, label = board.to(device), other.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(board, other)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss/bs

def val(device, model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (board, other, label) in enumerate(dataloader):
            board, other, label = board.to(device), other.to(device), label.to(device)
            output = model(board, other)
            loss = loss_fn(output, label)

            total_loss += loss.item()
    return total_loss/bs
