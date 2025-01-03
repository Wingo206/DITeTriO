print("importing")
import torch
from torch.cuda import is_initialized
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import torch.distributed as dist
import os
import argparse

import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split

from utilsNersc.logging import config_logging
from utilsNersc.distributed import init_workers
from model.tetr_dataset import load_dataset
from model.tetr_model import create_model
from model.lstm_training import train_lstm, val_lstm
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
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate brog")
parser.add_argument("--remove_last_frame", type=bool, default=False, help="Remove last frame inputs")
parser.add_argument("--lstm", type=bool, default=False, help="Use lstm model")
parser.add_argument("--seq_len", type=int, default=100, help="lstm sequence length for dataset")
parser.add_argument("--lstm_layers", type=int, default=1, help="lstm hidden layers")
parser.add_argument("--lstm_hidden_size", type=int, default=100, help="nodes in lstm hidden layers")
parser.add_argument("--lstm_dropout", type=float, default=0, help="dropout for lstm hidden nodes")
parser.add_argument("--sched_samp", type=bool, default=False, help="Enable scheduled sampling")
parser.add_argument("--ramp_epochs", type=int, default=False, help="amount of epochs to ramp up scheduled sampling")

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
lr = args.lr
epochs = args.epochs
bs = args.batch

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")


def init_weights(model):
    torch.manual_seed(1)
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight)
        nn.init.zeros_(model.bias)


shuffled = load_dataset(args)


def train(device, model, optimizer, loss_fn, dataloader):
    model.train()
    total_loss = 0
    total_samples = 0
    for batch in dataloader:
        # print(len(batch))
        board, other, label = batch
        new_board = board.reshape(board.shape[0], 1, 20, 10)
        new_other = other[:,:19]
        new_label = label[:,:8]
        board, other, label = new_board.to(device), new_other.to(device), new_label.to(device)
        optimizer.zero_grad()
        output = model(board, other)
        loss = loss_fn(output, label)
        loss.backward()
        avg_grad(model)
        optimizer.step()

        total_loss += loss.item()
        total_samples += 1

    return total_loss / total_samples

def avg_grad(model):
    for parameter in model.parameters():
        if type(parameter) is torch.Tensor:
            dist.all_reduce(parameter.grad.data,op=dist.ReduceOp.AVG)

def val(device, model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            board, other, label = batch
            new_board = board.reshape(board.shape[0], 1, 20, 10)
            new_other = other[:,:19]
            new_label = label[:,:8]
            board, other, label = new_board.to(device), new_other.to(device), new_label.to(device)
            output = model(board, other)
            loss = loss_fn(output, label)

            total_loss += loss.item()
            total_samples += 1 
    return total_loss / total_samples

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


    model = create_model(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    model.apply(init_weights)
    # loss_fn = nn.MSELoss().to(device)
    loss_fn = nn.BCELoss().to(device)

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
        if not args.lstm:
            train_loss = train(device, model, optimizer, loss_fn, train_dataset)
            val_loss = val(device, model, loss_fn, val_dataset)
        else:
            train_loss = train_lstm(device, model, optimizer, loss_fn, train_dataset, epoch)
            val_loss = val_lstm(device, model, loss_fn, val_dataset, epoch)

        # logging.info("Epoch %i Rank %i Train Loss: %f", epoch, rank, train_loss)
        # logging.info("Epoch %i Rank %i Val Loss: %f", epoch, rank, val_loss)

        losses = torch.tensor([train_loss, val_loss]).to(device)
        dist.all_reduce(losses, op=dist.ReduceOp.AVG)


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

    # cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    



if __name__ == "__main__":
    main()

