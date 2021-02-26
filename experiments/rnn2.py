import argparse, pickle
import numpy as np
import os
from os import path
from itertools import cycle

import torch
from torch import nn, optim

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dl.utils import *
from dl.models import *

dirname = os.path.dirname(os.path.abspath(__file__))


def cycle(seq):
    while True:
        for elem in seq:
            yield elem


def train(max_iter, model_name, batch_size=64, log_dir=None):
    train_dataloader = load_action(path.join('..', 'data', 'action_trainval', 'train.dat'), num_workers=4, crop=100)
    valid_dataloader = load_action(path.join('..', 'data', 'action_trainval', 'valid.dat'), num_workers=4, crop=100)

    train_dataloader_iterator = cycle(train_dataloader)
    valid_dataloader_iterator = cycle(valid_dataloader)

    # SeqModel = model_name
    model = eval(model_name)()  # .cuda()

    print(f"Num of parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

    log = None
    if log_dir is not None:
        from tensorboardX import SummaryWriter
        # from dl.core.utils import SummaryWriter
        log = SummaryWriter(log_dir)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss = nn.BCEWithLogitsLoss()

    for t in range(max_iter):
        batch = next(train_dataloader_iterator).float()  # .cuda().float()
        batch_inputs = batch[:, :, :-1]
        batch_outputs = batch[:, :, 1:]

        model.train()

        # zero the gradients (part of pytorch backprop)
        optimizer.zero_grad()

        # Compute the model output and loss (view flattens the input)
        model_outputs = model(batch_inputs)

        # Compute the loss
        t_loss_val = loss(model_outputs, batch_outputs) * 6

        # Compute the gradient
        t_loss_val.backward()

        # Update the weights
        optimizer.step()

        if t % 10 == 0:
            model.eval()

            valid_batch = next(valid_dataloader_iterator).float()  # .cuda().float()
            batch_inputs = batch[:, :, :-1]
            batch_outputs = batch[:, :, 1:]

            model_outputs = model(batch_inputs)

            v_loss_val = loss(model_outputs, batch_outputs) * 6

            print(f'[{t:5d}]  t_loss = {t_loss_val:1.6f}   v_loss_val = {v_loss_val:1.6f}')

            if log is not None:
                log.add_scalar('train/loss', t_loss_val, t)
                log.add_scalar('val/loss', v_loss_val, t)

    # Save the trained model
    name = 'rnn1' if model_name =='RNNModel1' else 'rnn2'
    torch.save(model.state_dict(), os.path.join(dirname, 'model_state', f'{name}.th'))  # Do NOT modify this line


scale_factor = 1.54


def test(iterations, model_name):
    # Load the model
    # SeqModel = model_name
    model = eval(model_name)()  # .cuda()
    name = 'rnn1' if model_name == 'RNNModel1' else 'rnn2'
    model.load_state_dict(torch.load(os.path.join(dirname, 'model_state', f'{name}.th')))
    model.eval()

    # Load the data
    # data = ActionDataset('valid.dat')
    data = ActionDataset(path.join('..', 'data', 'action_trainval', 'valid.dat'))

    loss = nn.BCEWithLogitsLoss(reduction='sum')

    loss_vals = []
    for i in range(iterations):
        print(f'{i:5d} iter')
        seq = torch.as_tensor(data[i]).float()
        pred = model.predictor()

        prob = None
        for i in range(seq.shape[-1]):
            if prob is not None:
                # Evaluate the prediction accuracy
                loss_vals.append(float(loss(prob, seq[:, i])))
            prob = pred(seq[:, i])
    print(f'Mean log-likelihood loss {np.mean(loss_vals)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-model', choices=['RNNModel1', 'RNNModel2'])
    parser.add_argument('-i_tr', '--max_iter', type=int, default=2000)
    parser.add_argument('-i_te', '--iterations', type=int, default=32)
    parser.add_argument('-l', '--log_dir')
    args = parser.parse_args()

    if args.train:
        print('Start training')
        train(args.max_iter, model_name=args.model, log_dir=args.log_dir)
        print('Training finished')
    if args.test:
        print('Testing')
        test(args.iterations, model_name=args.model)



