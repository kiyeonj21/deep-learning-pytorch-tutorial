import argparse, pickle
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch import nn, optim
from dl.utils import load
from dl.models import *


dirname = os.path.dirname(os.path.abspath(__file__))


class RegressLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_output, label):
        """
        model_output: size (N, 1)
        label:  size (N, 1)
        return value: scalar
        """
        return torch.mean((model_output-label)**2)


def label2onehot(label):
    # Transform a label of size (N,1) into a one-hot encoding of size (N,6)
    return (label[:,None] == torch.arange(6)[None]).float()


class OnehotLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_output, label):
        """
        model_output: size (N, 6)
        label:  size (N, 1)
        return value: scalar
        """
        return torch.mean(torch.sum((model_output - label2onehot(label))**2, dim=1))


class LlLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, model_output, label):
        """
        model_output: size (N, 6)
        label:  size (N, 1)
        return value: scalar
        """
        return self.loss(model_output, label)


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_output, label):
        """
        model_output: size (N, 6)
        label:  size (N, 1)
        return value: scalar
        """
        return torch.mean(torch.sum((torch.nn.functional.softmax(model_output, dim=1) - label2onehot(label))**2, dim=1))


def train(model_name, iterations, batch_size=64):
    train_inputs, train_labels = load(os.path.join('..', 'data', 'tux', 'tux_train.dat'))
    loss = eval(model_name.capitalize()+'Loss')()
    if model_name == 'regress':
        model = ScalarModel()
    else:
        model = VectorModel()

    # We use the ADAM optimizer with default learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for iteration in range(iterations):
        # Construct a mini-batch
        batch = np.random.choice(train_inputs.shape[0], batch_size)
        batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)
        batch_labels = torch.as_tensor(train_labels[batch], dtype=torch.long)

        if model_name == 'regress': # Regression expects float labels
            batch_labels = batch_labels.float()

        # zero the gradients (part of pytorch backprop)
        optimizer.zero_grad()

        # Compute the model output and loss (view flattens the input)
        loss_val = loss( model(batch_inputs.view(batch_size, -1)), batch_labels)

        # Compute the gradient
        loss_val.backward()

        # Update the weights
        optimizer.step()

        if iteration % 10 == 0:
            print(f'[{iteration:5d}] loss = {loss_val:1.6f}')

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'model_state', model_name + '.th')) # Do NOT modify this line


def test(model_name, iterations, batch_size=256):
    train_inputs, train_labels = load(os.path.join('..', 'data', 'tux', 'tux_valid.dat'))

    if model_name == 'regress':
        model = ScalarModel()
        pred = lambda x: x.detach().numpy().round().astype(int)
    else:
        model = VectorModel()
        pred = lambda x: np.argmax(x.detach().numpy(), axis=1)

    model.load_state_dict(torch.load(os.path.join(dirname, 'model_state', model_name + '.th')))

    accuracies = []
    for iteration in range(iterations):
        # Construct a mini-batch
        batch = np.random.choice(train_inputs.shape[0], batch_size)
        batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)

        pred_val = pred(model(batch_inputs.view(batch_size, -1)))
        accuracies.append(np.mean(pred_val == train_labels[batch]))
    print(f'Accuracy {np.mean(accuracies)} +- {np.std(accuracies)} ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-model', choices=['regress', 'onehot', 'LL', 'L2'])
    parser.add_argument('-i_tr', '--iterations_tr', type=int, default=10000)
    parser.add_argument('-i_te', '--iterations_te', type=int, default=10)
    args = parser.parse_args()

    if args.train:
        print (f'Start training {args.model}')
        train(args.model, args.iterations_tr)
        print ('Training finished')
    if args.test:
        print(f'Testing {args.model}')
        test(args.model, args.iterations_te)