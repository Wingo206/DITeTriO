import json
import shutil
import os
import subprocess
import argparse

# all files are in index, not all index are in files

parser = argparse.ArgumentParser(description="potential function trajectory")
parser.add_argument("--start", type=int, help="start index inclusive")
parser.add_argument("--end", type=int, help="end index exclusive")

args = parser.parse_args()


data_dir = "/global/cfs/cdirs/m4431/DITeTriO/data/top100/data"
processed_dir = "/global/cfs/cdirs/m4431/DITeTriO/data/processed_replays/players"
dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessor_build", "DITeTriOPreProcessor.dll")


with open(data_dir + "/index.json", "r") as f:
    index = json.load(f)

keys = index.keys()
print(len(keys))

for i in range(args.start, args.end):
# for i in range(2):
    print(i)

    filename = keys[i] + ".ttrm"
    if not os.path.exists(data_dir + "/" + filename):
        print(filename + " does not exist")
        continue
    print("processing " + filename)

    # get player names from the index
    game_info = index[keys[i]]
    players = [game_info[0]["username"], game_info[1]["username"]]
    print(players[0] + " vs " + players[1])

    # create directories for both players if don't exist
    for player in players:
        player_dir = os.path.join(processed_dir, player)
        if not os.path.exists(player_dir):
            os.makedirs(player_dir)

    # spawn subprocess to run the preprocessor
    command = ["dotnet", dll_path]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    inputs = [data_dir, keys[i], processed_dir, players[0], players[1]]
    input_string = " ".join(inputs)
    stdout, stderr = process.communicate(input=input_string)

    print(stdout)

    if stderr:
        print(stderr)












