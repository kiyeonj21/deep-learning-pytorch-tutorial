import argparse, pickle
import numpy as np
import os
from os import path
from itertools import cycle

import torch
from torch import nn, optim

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dl.utils import load_action, ActionDataset
from dl.models import RNNModel1

dirname = os.path.dirname(os.path.abspath(__file__))


def train(max_iter, batch_size=64, log_dir=None):
    train_dataloader = load_action(path.join('..', 'data', 'action_trainval', 'train.dat'), num_workers=4,
                            batch_size=batch_size, crop=100)
    valid_dataloader = load_action(path.join('..', 'data', 'action_trainval', 'valid.dat'), num_workers=4,
                            batch_size=batch_size, crop=100)

    train_dataloader_iterator = cycle(train_dataloader)
    valid_dataloader_iterator = cycle(valid_dataloader)

    model = RNNModel1()     # .cuda()


    log = None
    if log_dir is not None:
        from tensorboardX import SummaryWriter
        # from dl.core.utils import SummaryWriter
        log = SummaryWriter(log_dir)

    # If your model does not train well, you may swap out the optimizer or change the lr

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay=1e-4)

    loss = nn.BCEWithLogitsLoss()

    for t in range(max_iter):
        batch = next(train_dataloader_iterator).float()     #.cuda().float()
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

            print('[%5d]  t_loss = %f   v_loss_val = %f' % (t, t_loss_val, v_loss_val))

            if log is not None:
                log.add_scalar('train/loss', t_loss_val, t)
                log.add_scalar('val/loss', v_loss_val, t)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'model_state', 'rnn.th'))  # Do NOT modify this line


def test(iterations):
    # Load the model
    model = RNNModel1()     #.cuda()
    model.load_state_dict(torch.load(os.path.join(dirname, 'model_state', 'rnn.th')))
    model.eval()

    # Load the data
    # data = ActionDataset('valid.dat')
    data = ActionDataset(path.join('..', 'data', 'action_trainval', 'valid.dat'))

    loss = nn.BCEWithLogitsLoss(reduction='sum')

    loss_vals = []
    for i in range(iterations):
        print(i, 'iter')
        seq = torch.as_tensor(data[i]).float()  #.cuda().float()
        pred = model.predictor()

        prob = None
        for i in range(seq.shape[-1]):
            if prob is not None:
                # Evaluate the prediction accuracy
                loss_vals.append(float(loss(prob, seq[:, i])))
            prob = pred(seq[:, i])
    print('Mean log-likelihood loss', np.mean(loss_vals))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train',action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-i_tr', '--max_iter_tr', type=int, default=2000)
    parser.add_argument('-i_te', '--max_iter_te', type=int, default=32)
    parser.add_argument('-l', '--log_dir')
    args = parser.parse_args()
    if args.train:
        print('Start training')
        train(args.max_iter_tr, log_dir=args.log_dir)
        print('Training finished')
    if args.test:
        print('Testing')
        test(args.max_iter_te)