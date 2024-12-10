from scripts.visualizer import TetrisVisualizer
import torch.nn as nn
import pygame
import torch
from model.tetr_dataset import load_dataset
from model.tetr_model import combined_model, create_model
import argparse
from torch.utils.data import DataLoader
from argparse import Namespace
import numpy as np
import os
import re
import ast
import time

# dummy sound driver for pygame
os.environ["SDL_AUDIODRIVER"] = "dsp"
# os.environ["SDL_VIDEODRIVER"] = "x11"
# os.environ["DISPLAY"] = "localhost:10.0"


device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")


def load_model(model_dir):
    # get the information from the training log
    with open(os.path.join(model_dir, "summary.txt")) as f:
        content = f.read()
        model_args_str = content.splitlines()[0]

        model_args_str = model_args_str[10:-1]
        model_args_str = model_args_str.replace("=", ":")
        model_args_str = re.sub(r"(\w+):", r"'\1':", model_args_str)

        model_args_dict = ast.literal_eval(f"{{{model_args_str}}}")

        # need to remove first and last for some
        del model_args_dict["linears"][0]
        del model_args_dict["linears"][len(model_args_dict["linears"])-1]
        del model_args_dict["conv_channels"][0]

        # set scheduled sampling to false for evaluation
        model_args_dict["sched_samp"] = False

        model_args = Namespace(**model_args_dict)
        print(model_args)
    model = create_model(model_args).to(device)
    model.load_state_dict(torch.load( os.path.join(model_dir, "model.pt") ))

    return model

def process_batch_single(model, batch):
    """
    input model and batch
    returns vis state(s)
    """
    board, other, label = batch

    new_board = board.reshape(board.shape[0], 1, 20, 10)
    new_other = other[:,:19]
    new_label = label[:,:8]
    board, other, label = new_board.to(device), new_other.to(device), new_label.to(device)
    output = model(board, other)

    # update the visualizer state
    board.reshape(1, 200)

    # hack for extra value
    actual_outputs = (output.flatten().cpu().numpy() * 100).astype(int)
    actual_outputs += 100
    # actual_outputs = np.array([200, 0, 0, 0, 0, 0, 0, 0])
    print(label)
    print(np.round(output.cpu().numpy(), 2))

    loss_fn = nn.MSELoss()
    loss = loss_fn(output, label)
    print("loss is ", loss)

    vis_state = np.concatenate([ board.cpu().numpy().flatten(), other.cpu().numpy().flatten(), actual_outputs])
    # vis_state = np.concatenate([ board.cpu().numpy().flatten(), other.cpu().numpy().flatten(), np.array([0]), label.cpu().numpy().flatten()])
    vis_state = vis_state.astype(int)

    return [vis_state]


def process_batch_lstm(model, batch):
    """
    input model and batch (1, seq_len, stuff)
    returns vis state(s)
    """
    board, other, label = batch

    new_board = board
    new_other = other
    new_action = label[:,:-1,:]
    new_label = label[:,1:,:]
    board, other, action, label = new_board.to(device), new_other.to(device), new_action.to(device), new_label.to(device)
    output = model(board, other, action)

    # update the visualizer state

    # hack for extra value
    actual_outputs = (output.cpu().numpy() * 100).astype(int)
    actual_outputs += 100
    # actual_outputs = np.array([200, 0, 0, 0, 0, 0, 0, 0])
    print(label)
    print(np.round(output.cpu().numpy(), 2))

    loss_fn = nn.MSELoss()
    loss = loss_fn(output, label)
    print("loss is ", loss)

    _, seq_len, _ = board.shape
    vis_states = np.concatenate([ board.cpu().numpy(), other.cpu().numpy(), np.zeros(( 1, seq_len, 8 ), np.int32), actual_outputs], axis=2)
    vis_states = vis_states.astype(int)
    vis_states = vis_states.squeeze(0)
    print(vis_states.shape)
    print(vis_states)

    # convert to list
    vis_states_list = np.split(vis_states, vis_states.shape[0])
    for i in range(len(vis_states_list)):
        vis_states_list[i] = np.squeeze(vis_states_list[i])

    return vis_states_list

def process_batch(model, batch, args):
    if args.lstm:
        return process_batch_lstm(model, batch)
    else:
        return process_batch_single(model, batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DITeTriO Model")
    parser.add_argument("--train_data", help="path of data glob to use")
    parser.add_argument("--model_dir", help="directory with model in it")
    parser.add_argument("--remove_last_frame", type=bool, default=False, help="Remove last frame inputs")
    parser.add_argument("--lstm", type=bool, default=False, help="Use lstm model")
    parser.add_argument("--seq_len", type=int, default=100, help="lstm sequence length for dataset")

    args = parser.parse_args()

    model = load_model(args.model_dir)

    # get some data
    dataset = load_dataset(args)

    # initialize the visualizer
    visualizer = TetrisVisualizer()

    # run through all the data
    dataloader = DataLoader(dataset, batch_size = 1, shuffle=True)
    with torch.no_grad():
        for batch in dataloader:
            # process the batch into visualizer states
            vis_states = process_batch(model, batch, args)
            for vis_state in vis_states:
                print(vis_state)

                visualizer.update_state(vis_state)
                visualizer.draw_board()

                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:  # Allow quitting during wait
                            waiting = False
                            pygame.quit()
                            exit()
                        elif event.type == pygame.KEYDOWN:  # Detect key press
                            waiting = False




    

