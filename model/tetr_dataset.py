from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import pandas as pd

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


def load_dataset(path):
    datasets = []

    print("Loading Dataset")
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
    return shuffled

