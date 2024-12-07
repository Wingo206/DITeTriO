import os
import pygame
import time
import numpy as np
import pandas as pd
import sys
from io import StringIO
from scripts.visualizer import TetrisVisualizer
import argparse



class TetrEnvClient:
    def __init__(self, pipedir, replay_filepath, hook_fn):
        """
        hook_fn: input board array, output list of 8 inputs
        """
        
        # Create the named pipe if it doesn't exist
        cmd_pipe_name = os.path.join(pipedir, "cmdPipe")
        env_pipe_name = os.path.join(pipedir, "envPipe")
        if not os.path.exists(cmd_pipe_name):
            print("Making " + cmd_pipe_name)
            os.mkfifo(cmd_pipe_name)
        if not os.path.exists(env_pipe_name):
            print("Making " + env_pipe_name)
            os.mkfifo(env_pipe_name)

        # Open the pipe for writing
        print("Opening pipes")
        with open(cmd_pipe_name, "w") as cmd_pipe, open(env_pipe_name, "r") as env_pipe:
            print("Both pipes opened")

            # send filepath to open
            cmd_pipe.write(replay_filepath + "\n")
            cmd_pipe.flush()

            testval = False
            while (True):
                # infinitely read from the pipe
                readval = env_pipe.readline()
                if not readval:
                    break

                # sleep a bit to simulate model
                board_state = readval.rstrip()
                board_io = StringIO(board_state)
                df = pd.read_csv(board_io, header=None, index_col=False)
                board_state = df.to_numpy()[0]

                # Run the hook
                next_inputs = hook_fn(board_state)

                # send buttons
                next_inputs_str = " ".join(map(str, map(int, next_inputs))) + "\n"
                cmd_pipe.write(next_inputs_str)
                cmd_pipe.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipedir", help="directory to put the pipes in")
    parser.add_argument("--replay", help="path of replay to load from")
    args = parser.parse_args()

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

        out = np.random.rand(8) > 0.5
        return out

    TetrEnvClient(args.pipedir, args.replay, test_hook)
