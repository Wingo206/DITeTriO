import json
import shutil
import os

# all files are in index, not all index are in files

data_dir = "../data/top100/data/"

with open(data_dir + "index.json", "r") as f:
    index = json.load(f)

keys = index.keys()

for i in range(len(keys)):
    filename = keys[i] + ".ttrm"
    if not os.path.exists(data_dir + filename):
        print(filename + " does not exist")
        continue

    print(i)










