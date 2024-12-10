from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import pandas as pd
import argparse

gulagland = 30
class TetrDataset(Dataset):
    def __init__(self, df, args):
        if args.remove_last_frame:
            df.iloc[:, 211:219] = 0
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

        board_ten = torch.tensor(board.values, dtype=torch.float32)
        other_ten = torch.tensor(other.values, dtype=torch.float32)
        label_ten = torch.tensor(label.values, dtype=torch.float32)

        padded_other = F.pad(other_ten, (0, 181), "constant", 0)
        padded_label = F.pad(label_ten, (0,192), "constant", 0)

        return board_ten, padded_other, padded_label

# for use with LSTM
class TetrSequenceDataset(Dataset):
    def __init__(self, df, args):
        self.board = df.iloc[:,:200]
        self.other = df.iloc[:,200:211]  # don't load last frame info
        self.label = df.iloc[:,219:]
        self.seq_len = args.seq_len
    
    def __len__(self):
        return len(self.label) - self.seq_len - 1  # sub 1 for next frame
    
    def __getitem__(self, idx):
        seq_start = idx
        seq_end = idx + self.seq_len

        board = self.board.iloc[seq_start:seq_end]
        other = self.other.iloc[seq_start:seq_end]
        label = self.label.iloc[seq_start:seq_end+1]  # take care to offset the appropriate labels by 1

        board_ten = torch.tensor(board.values, dtype=torch.float32)
        other_ten = torch.tensor(other.values, dtype=torch.float32)
        label_ten = torch.tensor(label.values, dtype=torch.float32)

        return board_ten, other_ten, label_ten



def load_dataset(args):
    datasets = []

    print("Loading Dataset")
    files = glob.glob(args.train_data)

    if not args.lstm:
        for csv in tqdm(files, desc="Reading Files"):
            data = pd.read_csv(csv) #input csv here
            data = data[:-gulagland]
            data[["moveleft","moveright","softdrop","rotate_cw","rotate_ccw","rotate_180","harddrop","hold"]] = data[["moveleft","moveright","softdrop","rotate_cw","rotate_ccw","rotate_180","harddrop","hold"]].shift(-1)
            data = data.dropna()
            data = data.sample(frac=1)
            data = data.reset_index(drop=True)
            data["can_hold"] = data["can_hold"].astype(int)
            data.iloc[:,0:200] = data.iloc[:,0:200].applymap(lambda x: 1 if x != 0 else x)
            temp = TetrDataset(data, args)
            datasets.append(temp)

        shuffled = ConcatDataset(datasets)
        print("Done Loading Dataset")
        return shuffled
    else:
        print("hello")
        # load sequential datasets
        for csv in tqdm(files, desc="Reading Files"):
            data = pd.read_csv(csv) #input csv here
            data = data.dropna()
            # skip if too little data
            if len(data) < args.seq_len:
                print("Skipping")
                continue
            data = data.reset_index(drop=True)
            data["can_hold"] = data["can_hold"].astype(int)
            data.iloc[:,0:200] = data.iloc[:,0:200].applymap(lambda x: 1 if x != 0 else x)
            temp = TetrSequenceDataset(data, args)
            datasets.append(temp)

        out = ConcatDataset(datasets)
        print("Done Loading Dataset")
        return out



# for testing
"""
python -m model.tetr_dataset --train_data "data/processed_replays/players/linustechtips/*.csv" --lstm True --seq_len 100
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DITeTriO model")
    parser.add_argument("--train_data", help="path to data to train on")
    parser.add_argument("--remove_last_frame", type=bool, default=False, help="Remove last frame inputs")
    parser.add_argument("--lstm", type=bool, default=False, help="Use lstm model if true")
    parser.add_argument("--seq_len", type=int, default=100, help="LSTM sequence length")

    args = parser.parse_args()
    if args.train_data.startswith('"') and args.train_data.endswith('"'):
        args.train_data = args.train_data[1:-1]

    dataset = load_dataset(args)

    print(dataset[0])

