import torch.nn as nn
import numpy as np
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
    def __init__(self, args, first_layer_extra):
        super(prob_model,self).__init__()

        args.linears.insert(0, first_layer_extra)  # number of input nodes
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

    def forward(self, x):
        return self.seq1(x)
    
class combined_model(nn.Module):
    def __init__(self, args):
        super(combined_model,self).__init__()

        self.board_model = board_model(args)
        # run the board model once to see the output size
        input_size = (1, 1, 20, 10)
        dummy_input = torch.randn(input_size)
        dummy_output = self.board_model(dummy_input)

        first_layer_extra = 19

        self.prob_model = prob_model(args, dummy_output.shape[1] + first_layer_extra)


    def forward(self, x, new_data):
        x = self.board_model(x)
        x = torch.cat((x, new_data),dim=1)
        return self.prob_model(x)

class lstm_model(nn.Module):
    def __init__(self, args):
        super(lstm_model, self).__init__()

        self.board_model = board_model(args)
        # run the board model once to see the output size
        input_size = (1, 1, 20, 10)
        dummy_input = torch.randn(input_size)
        dummy_output = self.board_model(dummy_input)

        self.lstm = nn.LSTM(8, args.lstm_hidden_size, num_layers=args.lstm_layers, batch_first=True, dropout=args.lstm_dropout)

        first_layer_extra = 11 + args.lstm_hidden_size
        self.prob_model = prob_model(args, dummy_output.shape[1] + first_layer_extra)

    def forward(self, board, other, action):
        lstm_out, _ = self.lstm(action)

        # reshape the board from (batch, seq_len, 200) into (batch*seq_len, 1, 20, 10)
        batch_size, seq_len, _ = board.shape
        reshaped = board.view(batch_size * seq_len, 1, 20, 10)
        board_out = self.board_model(reshaped)
        # reshape back into (batch, seq_len, flattened_output)
        board_out = board_out.view(batch_size, seq_len, -1)

        combined = torch.cat((board_out, other, lstm_out), dim=2)

        outputs = self.prob_model(combined)
        return outputs

class lstm_scheduled_sample_model(lstm_model):
    def __init__(self, args):
        super(lstm_scheduled_sample_model, self).__init__(args)
        
        start_prob = 0.1
        end_prob = 0.9
        total_epochs = args.ramp_epochs
        # linear ramp up
        self.schedule_fn = lambda epoch: min(end_prob, start_prob + (epoch / total_epochs) * (end_prob - start_prob))

    def forward(self, board, other, action, epoch):
        # do forward but with scheduled sampling

        # precompute the board model for all in the sequence
        # reshape the board from (batch, seq_len, 200) into (batch*seq_len, 1, 20, 10)
        batch_size, seq_len, _ = board.shape
        reshaped = board.view(batch_size * seq_len, 1, 20, 10)
        board_out = self.board_model(reshaped)
        # reshape back into (batch, seq_len, flattened_output)
        board_out = board_out.view(batch_size, seq_len, -1)

        # initial lstm values
        actual_action = action[:, 0:1, :].clone()
        h, c = None, None
        predictions = []  # will be used for the output
            
        for t in range(seq_len):
            # Compute each step 1 at a time
            if h is None:
                lstm_out, (h, c) = self.lstm(actual_action[:, t:t+1, :])  # first one, don't input h,c
            else:
                lstm_out, (h, c) = self.lstm(actual_action[:, t:t+1, :], (h, c))
            combined = torch.cat((board_out[:, t:t+1, :], other[:, t:t+1, :], lstm_out), dim=2)
            pred_action = self.prob_model(combined)
            predictions.append(pred_action)  # store prediction, add dimension for seq_len

            # sample to get actual action for this timestep if not the last one
            if t + 1 < seq_len:
                sample_mask = torch.rand(batch_size, 1, device=pred_action.device) < self.schedule_fn(epoch)
                sample_mask = sample_mask.view(batch_size, 1, 1).expand_as(action[:, t+1:t+2, :])
                # use model predictions for t+1 action
                # detach because the sampled action is not differentiable, we shouldn't use this for grad calc
                sampled_action = torch.where(
                    sample_mask,
                    torch.bernoulli(pred_action).detach(), 
                    action[:, t+1:t+2, :]
                    )

                # append to the actual action, no inplace setting to avoid gradient issues
                actual_action = torch.cat([ actual_action, sampled_action], dim=1)


        # concat the predictions into one tensor along sequence length
        predictions_ten = torch.cat(predictions, dim=1)
        return predictions_ten


def create_model(args):
    if args.lstm:
        if args.sched_samp:
            return lstm_scheduled_sample_model(args)
        else:
            return lstm_model(args)
    else:
        return combined_model(args)
