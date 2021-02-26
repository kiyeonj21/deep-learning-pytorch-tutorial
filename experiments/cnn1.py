import argparse, pickle
import numpy as np
import os
import torch
from torch import nn, optim
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dl.utils import load
from dl.models import *

dirname = os.path.dirname(os.path.abspath(__file__))


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(outputs)
    return outputs_idx.eq(labels.float()).float().mean()


def train(iterations, batch_size=64, log_dir=None):
    train_inputs, train_labels = load(os.path.join('..', 'data', 'tux', 'tux_train.dat'))
    model = ConvNetModel1()
    log = None
    if log_dir is not None:
        from tensorboardX import SummaryWriter
        # from dl.utils import SummaryWriter
        log = SummaryWriter(log_dir)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()

    for iteration in range(iterations):
        model.train()
        # Construct a mini-batch
        batch = np.random.choice(train_inputs.shape[0], batch_size)
        batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)
        batch_inputs = batch_inputs.permute(0, 3, 1, 2)
        batch_labels = torch.as_tensor(train_labels[batch], dtype=torch.long)

        # zero the gradients (part of pytorch backprop)
        optimizer.zero_grad()

        # Compute the model output and loss (view flattens the input)
        model_outputs = model(batch_inputs)
        t_loss_val = loss(model_outputs, batch_labels)
        t_acc_val = accuracy(model_outputs, batch_labels)

        # Compute the gradient
        t_loss_val.backward()

        # Update the weights
        optimizer.step()

        if iteration % 10 == 0:
            model.eval()

            print(f'[{iteration:5d}] loss = {t_loss_val:1.6f} acc = {t_acc_val:1.6f}')
            if log is not None:
                log.add_scalar('train/loss', t_loss_val, iteration)
                log.add_scalar('train/acc', t_acc_val, iteration)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'model_state', 'convnet1.th'))  # Do NOT modify this line


def test(iterations, batch_size=256):
    # train_inputs, train_labels = load(os.path.join('tux_valid.dat'))
    train_inputs, train_labels = load(os.path.join('..', 'data', 'tux', 'tux_valid.dat'))

    model = ConvNetModel1()
    pred = lambda x: np.argmax(x.detach().numpy(), axis=1)

    model.load_state_dict(torch.load(os.path.join(dirname, 'model_state', 'convnet1.th')))
    model.eval()

    accuracies = []
    for iteration in range(iterations):
        # Construct a mini-batch
        batch = np.random.choice(train_inputs.shape[0], batch_size)
        batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)
        batch_inputs = batch_inputs.permute(0, 3, 1, 2)
        pred_val = pred(model(batch_inputs.view(batch_size, 3, 64, 64)))
        accuracies.append(np.mean(pred_val == train_labels[batch]))
    print(f'Accuracy {np.mean(accuracies):.6f} +- {np.std(accuracies) / np.sqrt(len(accuracies)):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-i_tr', '--iterations_tr', type=int, default=10000)
    parser.add_argument('-i_te', '--iterations_te', type=int, default=10)
    parser.add_argument('-l', '--log_dir')
    args = parser.parse_args()

    if args.train:
        print('Start training')
        train(args.iterations_tr, log_dir=args.log_dir)
        print('Training finished')
    if args.test:
        print('Testing')
        test(args.iterations_te)
