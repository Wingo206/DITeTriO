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
board_lr = 1e-3
prob_lr = 1e-3


board_model = nn.Sequential(
  nn.Conv2d(C, channel_1, (kernel_size_1, kernel_size_1), padding = pad_size_1),
  nn.LeakyReLU(negative_slope=leakySlope),
  nn.Conv2d(channel_1, channel_2, (kernel_size_2, kernel_size_2), padding = pad_size_2),
  nn.LeakyReLU(negative_slope=leakySlope),
  nn.Conv2d(channel_2, channel_3, (kernel_size_3, kernel_size_3), padding = pad_size_3),
  nn.LeakyReLU(negative_slope=leakySlope),
  nn.Flatten()
)

prob_model = nn.Sequential(
    nn.Linear(channel_3*H*W+21, hidden_layer_1),
    nn.Linear(hidden_layer_1, hidden_layer_2),
    nn.Linear(hidden_layer_2, output)
)

board_optimizer = optim.adam(board_model.parameters(), board_lr)
prob_optimizer = optim.adam(prob_model.parameters(), prob_lr)

#need to write a dataloader of some sort

def train(device, board_model, prob_model, board_optimizer, prob_optimizer):
    board_model.train()
    prob_model.train()