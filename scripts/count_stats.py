import json
import os

data_dir = "/global/cfs/cdirs/m4431/DITeTriO/data/top100/data"

with open(data_dir + "/index.json", "r") as f:
    index = json.load(f)

keys = index.keys()
print(len(keys))

total_pieces = 0

for i in range(len(keys)):
    filename = keys[i] + ".ttrm"
    if not os.path.exists(data_dir + "/" + filename):
        print(filename + " does not exist")
        continue

    info = index[keys[i]]
    total_pieces += info[0]["piecesplaced"]
    total_pieces += info[1]["piecesplaced"]

print(total_pieces)

