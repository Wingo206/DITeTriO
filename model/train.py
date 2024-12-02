import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import random
import logging
import torch.distributed as dist
import os
import torch.nn.functional as F
from torch.utils.data import Sampler


# from utilsNersc.logging import config_logging
# from utilsNersc.distributed import init_workers

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

        self.board_model = board_model()
        self.prob_model = prob_model()

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

model = combined_model().to(device)
optimizer = optim.Adam(model.parameters(), lr)
model.apply(init_weights)
loss_fn = nn.MSELoss()

#need to write a dataloader of some sort

class TetrDataset(Dataset):
    def __init__(self, df):
        self.board = df.iloc[:,0:199]
        self.other = df.iloc[:,200:220]
        self.label = df.iloc[:,221:228]
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        # print(idx)
        board = self.board.iloc[idx]
        other = self.other.iloc[idx]
        label = self.label.iloc[idx]

        # print(board.head())

        board_ten = torch.tensor(board.values, dtype=torch.int32)
        other_ten = torch.tensor(other.values, dtype=torch.int32)
        label_ten = torch.tensor(label.values, dtype=torch.int32)
        
        # print(board_ten)
        # print(other_ten)
        # print(label_ten)

        return board_ten, other_ten, label_ten

gulagland = 30
data = pd.read_csv("data/processed_replays/test2_0.csv") #input csv here
data = data[:-gulagland]
data[["moveleft","moveright","softdrop","rotate_cw","rotate_ccw","rotate_180","harddrop","hold"]] = data[["moveleft","moveright","softdrop","rotate_cw","rotate_ccw","rotate_180","harddrop","hold"]].shift(-1)
data = data.dropna()
data = data.sample(frac=1)
data = data.reset_index(drop=True)
data["can_hold"] = data["can_hold"].astype(int)
data.iloc[:,0:199] = data.iloc[:,0:199].map(lambda x: 1 if x != 0 else x)
shuffled = TetrDataset(data)


def cust_coll(batch):
    print(batch)
    board = [item[0] for item in batch]
    other = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return board, other, label


random.seed(1)
train_data = shuffled[0:int(0.9*len(shuffled))]
val_data = shuffled[int(0.9*len(shuffled)):]

train_dataset = DataLoader(train_data, batch_size = bs,shuffle = True, collate_fn= cust_coll)
val_dataset = DataLoader(val_data, batch_size = bs, shuffle = True, collate_fn= cust_coll)


def train(device, model, optimizer, loss_fn, dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        print(len(batch))
        board, other, label = batch
        print(board.shape)
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
        for board, other, label in dataloader:
            board, other, label = board.to(device), other.to(device), label.to(device)
            output = model(board, other)
            loss = loss_fn(output, label)

            total_loss += loss.item()
    return total_loss/bs

def main():

    # rank, n_ranks = init_workers()

    # output_dir = "$SCRATCH/tetris/output"
    # output_dir = os.path.expandvars(output_dir)
    # os.makedirs(output_dir, exist_ok=True)

    # log_file = (os.path.join(output_dir, 'out_%i.log' % rank)
    #             if output_dir is not None else None)
    # config_logging(verbose=True, log_file=log_file)

    # logging.info('Initialized rank %i out of %i', rank, n_ranks)

    for epoch in range(epochs):
        train_loss = train(device, model, optimizer, loss_fn, train_dataset)
        val_loss = val(device, model, optimizer, loss_fn, val_dataset)
        print("Epoch %i Train Loss: %f", epoch, train_loss)
        # logging.info("Epoch %i Train Loss: %f", epoch, train_loss)
        # logging.info("Epoch %i Val Loss: %f", epoch, val_loss)

if __name__ == "__main__":
    main()

