import os
import pygame
import time
import numpy as np
import sys
from scripts.visualizer import TetrisVisualizer
from scripts.evaluate_model import load_model
from scripts.tetr_env_client import TetrEnvClient
import argparse
import torch

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipedir", help="directory to put the pipes in")
    parser.add_argument("--replay", help="path of replay to load from")
    parser.add_argument("--model_dir", help="directory with model in it")
    parser.add_argument("--remove_last_frame", type=bool, default=False, help="Remove last frame inputs")
    parser.add_argument("--lstm", type=bool, default=False, help="Use lstm model")
    parser.add_argument("--seq_len", type=int, default=100, help="lstm sequence length for dataset")
    args = parser.parse_args()

    model = load_model(args.model_dir)
    model.eval()

    os.environ["SDL_AUDIODRIVER"] = "dsp"
    vis = TetrisVisualizer() 

    global testval
    testval = False
    def test_hook(board_state):
        global testval
        vis.update_state(board_state)
        vis.draw_board()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit()

        time.sleep(0.2)

        # run the model on the inputs
        board = board_state[:200].reshape(1, 1, 20, 10)
        other = board_state[200:219]
        label = board_state[219:]

        # remove last frame inputs
        if args.remove_last_frame:
            other[11:19] = 0

        # hard cap botch
        other = np.clip(other, -20, 10)

        new_board = torch.tensor(board.astype(np.int32), dtype=torch.float32).to(device)
        new_other = torch.tensor(other.astype(np.int32), dtype=torch.float32).to(device)
        new_other = new_other.unsqueeze(0)


        output = model(new_board, new_other)
        outlist = output.cpu().detach().numpy()[0]
        print("model output: ", outlist)
        
        # sample the inputs to determine the buttons
        choices = np.random.rand(8)
        print("choices: ", choices)

        output = choices < outlist

        #botch: overwrite if more than 10 frames of holding
        lastframetimes = other[11:19]
        output = np.logical_and(output, lastframetimes > -10)
        print(output)

        output = output.tolist()
        return output



    previous_actions = []  # list of tensors
    previous_actions.append(torch.zeros(8).to(device))
    boards = []  # list of tensors
    others = []
    # botch
    # for i in range(99):
    #     previous_actions.append(torch.rand(8).to(device))
    #     boards.append(torch.zeros(200).to(device))
    #     others.append(torch.zeros(11).to(device))

    def lstm_hook(board_state):
        vis.update_state(board_state)
        vis.draw_board()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit()

        time.sleep(0.2)

        # run the model on the inputs
        board = board_state[:200]
        other = board_state[200:211]
        label = board_state[219:]

        new_board = torch.tensor(board.astype(np.int32), dtype=torch.float32).to(device)
        new_other = torch.tensor(other.astype(np.int32), dtype=torch.float32).to(device)
        boards.append(new_board)
        others.append(new_other)
        if len(boards) > args.seq_len:
            del boards[0]
            del others[0]
        
        # turn the lists into tensors
        boards_ten = torch.stack(boards, dim=0)
        boards_ten = boards_ten.unsqueeze(0)  # add the batch_size dimension
        others_ten = torch.stack(others, dim=0)
        others_ten = others_ten.unsqueeze(0)
        actions_ten = torch.stack(previous_actions, dim=0)
        actions_ten = actions_ten.unsqueeze(0)

        # run the model
        output = model(boards_ten, others_ten, actions_ten)

        print(output.shape)
        # get the next action
        next_action_ten = output[0,-1,:]
        previous_actions.append(next_action_ten)
        if len(previous_actions) > args.seq_len:
            del previous_actions[0]

        print(next_action_ten.shape)
        outlist = next_action_ten.cpu().detach().numpy()
        print("model output: ", np.round(outlist, 2))
        
        # sample the inputs to determine the buttons
        choices = np.random.rand(8)
        print("choices: ", choices)
        output = choices < outlist

        # output = outlist > 0.5


        # lastframetimes = board_state[211:219]
        # output = np.logical_and(output, lastframetimes > -10)

        output = output.tolist()
        return output



    TetrEnvClient(args.pipedir, args.replay, lstm_hook)
