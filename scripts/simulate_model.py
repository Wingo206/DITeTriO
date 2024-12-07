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


    TetrEnvClient(args.pipedir, args.replay, test_hook)
