import torch.nn as nn
import torch

leakySlope = 1e-2

class board_model(nn.Module):
    def __init__(self, args):
        super(board_model,self).__init__()

        layers = []
        args.conv_channels.insert(0, 1)
        for i in range(len(args.conv_kernels)):
            k = args.conv_kernels[i]
            layers.append(nn.Conv2d(args.conv_channels[i], args.conv_channels[i+1], (k, k), padding = args.conv_padding[i]))
            layers.append(nn.LeakyReLU(negative_slope=leakySlope))

        layers.append(nn.Flatten())

        self.seq1 = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.seq1(x)

class prob_model(nn.Module):
    def __init__(self, args, cnn_features_size):
        super(prob_model,self).__init__()

        args.linears.insert(0, cnn_features_size + 19)  # number of input nodes
        args.linears.append(8)  # number of output nodes
        layers = []
        for i in range(len(args.linears) - 1):
            layers.append(nn.Linear(args.linears[i], args.linears[i+1]))
            if i < len(args.linears) - 2:  # only add relu between layers, not at the end
                layers.append(nn.LeakyReLU(negative_slope=leakySlope))

            if i == len(args.dropouts):
                continue
            if args.dropouts[i] > 0:
                layers.append(nn.Dropout(p=args.dropouts[i]))

        layers.append(nn.Sigmoid())

        self.seq1 = nn.Sequential(*layers)

    def forward(self, x, new_data):
        x = torch.cat((x, new_data),dim=1)
        return self.seq1(x)
    
class combined_model(nn.Module):
    def __init__(self, args):
        super(combined_model,self).__init__()

        self.board_model = board_model(args)
        # run the board model once to see the output size
        input_size = (1, 1, 10, 20)
        dummy_input = torch.randn(input_size)
        dummy_output = self.board_model(dummy_input)

        self.prob_model = prob_model(args, dummy_output.shape[1])

    def forward(self, x, new_data):
        x = self.board_model(x)
        return self.prob_model(x, new_data)
