import torch
import torch.distributed as dist

def train_lstm(device, model, optimizer, loss_fn, dataloader, epoch):
    model.train()
    total_loss = 0
    total_samples = 0
    for batch in dataloader:
        # each one: (batch, seq len, data dimension)
        board, other, label = batch

        new_board = board
        new_other = other
        new_action = label[:,:-1,:]
        new_label = label[:,1:,:]
        board, other, action, label = new_board.to(device), new_other.to(device), new_action.to(device), new_label.to(device)

        optimizer.zero_grad()
        output = model(board, other, action, epoch)

        loss = loss_fn(output, label)
        loss.backward()
        avg_grad(model)
        optimizer.step()

        total_loss += loss.item()
        total_samples += 1

    return total_loss / total_samples

def val_lstm(device, model, loss_fn, dataloader, epoch):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            board, other, label = batch
            new_board = board
            new_other = other
            new_action = label[:,:-1,:]
            new_label = label[:,1:,:]
            board, other, action, label = new_board.to(device), new_other.to(device), new_action.to(device), new_label.to(device)
            output = model(board, other, action, epoch)
            loss = loss_fn(output, label)

            total_loss += loss.item()
            total_samples += 1
    return total_loss / total_samples

def avg_grad(model):
    for parameter in model.parameters():
        if type(parameter) is torch.Tensor:
            dist.all_reduce(parameter.grad.data,op=dist.ReduceOp.AVG)
