print("importing")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import logging
import torch.distributed as dist
import os
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import glob
from torch.utils.data import Subset
from tqdm import tqdm
import argparse
from torchinfo import summary

import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split

from utilsNersc.logging import config_logging
from utilsNersc.distributed import init_workers
print("done importing")

parser = argparse.ArgumentParser(description="Train DITeTriO model")
parser.add_argument("--train_data", help="path to data to train on")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--batch", type=int, default=10000, help="Batch size")
parser.add_argument("--output_dir", help="path to save model and logs in")
parser.add_argument("--conv_channels", type=int, nargs="+", help="number of channels for each convolutional layer")
parser.add_argument("--conv_kernels", type=int, nargs="+", help="size of convolutional kernels")
parser.add_argument("--conv_padding", type=int, nargs="+", help="size of maxpooling kernels")
parser.add_argument("--linears", type=int, nargs="+", help="number of nodes for each linear layer")
parser.add_argument("--dropouts", type=float, nargs="+", help="dropout percentages for linear layers")

args = parser.parse_args()
# remove quotes from train data and output dir
if args.train_data.startswith('"') and args.train_data.endswith('"'):
    args.train_data = args.train_data[1:-1]
print(args)

C, H, W = 1, 20, 10

# default convolutions
if not args.conv_channels:
    args.conv_channels = [32, 16, 8]

if not args.conv_kernels:
    args.conv_kernels = [9, 7, 5]

if not args.conv_padding:
    args.conv_padding = [int((x-1)/2) for x in args.conv_channels]

if not args.linears:
    args.linears = [100, 100]

if not args.dropouts:
    args.dropouts = [0 for _ in args.linears]

output = 8

leakySlope = 1e-2
lr = 1e-3
epochs = args.epochs
bs = args.batch

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

class board_model(nn.Module):
    def __init__(self):
        super(board_model,self).__init__()

        layers = []
        args.conv_channels.insert(0, 1)
        for i in range(len(args.conv_kernels)):
            k = args.conv_kernels[i]
            layers.append(nn.Conv2d(args.conv_channels[i], args.conv_channels[i+1], (k, k), padding = args.conv_padding[i]))
            layers.append(nn.LeakyReLU(negative_slope=leakySlope))

        layers.append(nn.Flatten())

        self.seq1 = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.seq1(x)

class prob_model(nn.Module):
    def __init__(self, cnn_features_size):
        super(prob_model,self).__init__()

        args.linears.insert(0, cnn_features_size + 18)  # number of input nodes
        args.linears.append(8)  # number of output nodes
        layers = []
        for i in range(len(args.linears) - 1):
            layers.append(nn.Linear(args.linears[i], args.linears[i+1]))
            layers.append(nn.LeakyReLU(negative_slope=leakySlope))

            if i == len(args.dropouts):
                continue
            if args.dropouts[i] > 0:
                layers.append(nn.Dropout(p=args.dropouts[i]))

        self.seq1 = nn.Sequential(*layers)

    def forward(self, x, new_data):
        x = torch.cat((x, new_data),dim=1)
        return self.seq1(x)
    
class combined_model(nn.Module):
    def __init__(self):
        super(combined_model,self).__init__()

        self.board_model = board_model()
        # run the board model once to see the output size
        input_size = (1, 1, 10, 20)
        dummy_input = torch.randn(input_size)
        dummy_output = self.board_model(dummy_input)

        self.prob_model = prob_model(dummy_output.shape[1])

    def forward(self, x, new_data):
        x = self.board_model(x)
        return self.prob_model(x, new_data)

def init_weights(model):
    torch.manual_seed(1)
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight)
        nn.init.zeros_(model.bias)



#need to write a dataloader of some sort

class TetrDataset(Dataset):
    def __init__(self, df):
        self.board = df.iloc[:,:200]
        self.other = df.iloc[:,200:219]
        self.label = df.iloc[:,219:]
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        # print(idx)
        board = self.board.iloc[idx]
        other = self.other.iloc[idx]
        label = self.label.iloc[idx]

        # print(other.head())
        # print(label.head())

        board_ten = torch.tensor(board.values, dtype=torch.float32)
        other_ten = torch.tensor(other.values, dtype=torch.float32)
        label_ten = torch.tensor(label.values, dtype=torch.float32)

        padded_other = F.pad(other_ten, (0, 181), "constant", 0)
        padded_label = F.pad(label_ten, (0,192), "constant", 0)
        
        # print(board_ten)
        # print(other_ten)
        # print(label_ten)

        return board_ten, padded_other, padded_label

gulagland = 30
datasets = []

print("Loading Dataset")
path = args.train_data
files = glob.glob(path)

for csv in tqdm(files, desc="Reading Files"):
    data = pd.read_csv(csv) #input csv here
    data = data[:-gulagland]
    data[["moveleft","moveright","softdrop","rotate_cw","rotate_ccw","rotate_180","harddrop","hold"]] = data[["moveleft","moveright","softdrop","rotate_cw","rotate_ccw","rotate_180","harddrop","hold"]].shift(-1)
    data = data.dropna()
    data = data.sample(frac=1)
    data = data.reset_index(drop=True)
    data["can_hold"] = data["can_hold"].astype(int)
    data.iloc[:,0:200] = data.iloc[:,0:200].applymap(lambda x: 1 if x != 0 else x)
    temp = TetrDataset(data)
    datasets.append(temp)

shuffled = ConcatDataset(datasets)
print("Done Loading Dataset")


def train(device, model, optimizer, loss_fn, dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # print(len(batch))
        board, other, label = batch
        new_board = board.reshape(board.shape[0], 1, 10, 20)
        new_other = other[:,:18]
        new_label = label[:,:8]
        board, other, label = new_board.to(device), new_other.to(device), new_label.to(device)
        optimizer.zero_grad()
        output = model(board, other)
        loss = loss_fn(output, label)
        loss.backward()
        avg_grad(model)
        optimizer.step()

        total_loss += loss.item()
    return total_loss/len(dataloader.sampler)

def avg_grad(model):
    for parameter in model.parameters():
        if type(parameter) is torch.Tensor:
            dist.all_reduce(parameter.grad.data,op=dist.reduce_op.AVG)

def val(device, model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            board, other, label = batch
            new_board = board.reshape(board.shape[0], 1, 10, 20)
            new_other = other[:,:18]
            new_label = label[:,:8]
            board, other, label = new_board.to(device), new_other.to(device), new_label.to(device)
            output = model(board, other)
            loss = loss_fn(output, label)

            total_loss += loss.item()
    return total_loss/len(dataloader.sampler)

def main():


    rank, n_ranks = init_workers("nccl")

    # output_dir = "$SCRATCH/tetris/output"
    output_dir = args.output_dir
    output_dir = os.path.expandvars(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_file = (os.path.join(output_dir, 'out_%i.log' % rank)
                if output_dir is not None else None)
    config_logging(verbose=False, log_file=log_file)

    logging.info('Initialized rank %i out of %i', rank, n_ranks)

    gen  = torch.Generator().manual_seed(1)
    trainset, valset = random_split(shuffled, [0.9,0.1], generator=gen)
    train_sampler = DistributedSampler(trainset, num_replicas=n_ranks, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valset, num_replicas=n_ranks, rank=rank, shuffle=True)
                    
    train_dataset = DataLoader(trainset, batch_size = int(bs/n_ranks), sampler=train_sampler)
    val_dataset = DataLoader(valset, batch_size = int(bs/n_ranks), sampler=val_sampler)


    model = combined_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    model.apply(init_weights)
    loss_fn = nn.MSELoss().to(device)

    # log arguments
    logging.info(args)
    if rank == 0:
        with open(os.path.join(output_dir, "summary.txt"), "w+") as f:
            f.write(str(args) + "\n")
            for layer in model.children():
                logging.info(layer)
                f.write(str(layer) + "\n")

    train_loss_data = []
    val_loss_data = []

    for epoch in range(epochs):
        train_loss = train(device, model, optimizer, loss_fn, train_dataset)
        val_loss = val(device, model, loss_fn, val_dataset)
        logging.info("Epoch %i Rank %i Train Loss: %f", epoch, rank, train_loss)
        logging.info("Epoch %i Rank %i Val Loss: %f", epoch, rank, val_loss)

        losses = torch.tensor([train_loss, val_loss]).to(device)
        dist.all_reduce(losses)


        if rank == 0:
            losses = losses.cpu()
            train_loss_data.append(losses[0])
            val_loss_data.append(losses[1])
            logging.info("Epoch %i Average Losses: %f, %f", epoch, losses[0], losses[1])


            plt.plot(range(len(train_loss_data)), train_loss_data, 'b-', label='Training Loss')
            plt.plot(range(len(val_loss_data)), val_loss_data, 'r-', label='Validation Loss')
            plt.yscale('log')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(output_dir, "loss.png"), dpi=200)

            plt.clf()
            logging.info("saving model")
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            logging.info("done saving model")



if __name__ == "__main__":
    main()

