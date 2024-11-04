from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

optimizer = optim.Adam(prob_model.parameters(), lr)
model = combined_model.to(device)
loss_fn = nn.CrossEntropyLoss()

#need to write a dataloader of some sort

def train(device, model, optimizer, loss_fn):
    model.train()