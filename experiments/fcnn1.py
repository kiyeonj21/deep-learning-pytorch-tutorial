import argparse, pickle
import numpy as np
import os
from os import path
from itertools import cycle

import torch
from torch import nn, optim
from torchvision import transforms

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dl.utils import load_tux
from dl.models import *

import matplotlib
import matplotlib.pyplot as plt

dirname = os.path.dirname(os.path.abspath(__file__))


def one_hot(x, n=6):
    return x.view(-1, 1) == torch.arange(6, dtype=x.dtype, device=x.device)[None]


def J(M):
    return (M.diag() / (M.sum(dim=0) + M.sum(dim=1) - M.diag() + 1e-5)).mean()


def CM(outputs, true_labels):
    return torch.matmul(one_hot(outputs).t().float(), one_hot(true_labels).float())


def iou(outputs, true_labels):
    return J(CM(outputs, true_labels))


train_class_loss_weights = 1 + np.power(
    np.array(
        [0.6608633516136736, 0.004526087574279228, 0.016993887228547416, 7.296233198007641e-05, 0.0035574906588831134,
         0.3139862205926366]),
    -0.9,
)
val_class_loss_weights = 1 + np.power(
    np.array(
        [0.7145916062958386, 0.004292437495017538, 0.016724738296197385, 5.7827696508290815e-05, 0.003745456617705676,
         0.26058793359873245]),
    -0.9,
)


def train(max_iter, batch_size=64, log_dir=None):
    train_dataloader = load_tux(path.join('..', 'data', 'fc_trainval', 'train'), num_workers=4, crop=64)
    valid_dataloader = load_tux(path.join('..', 'data', 'fc_trainval', 'valid'), num_workers=4)
    train_dataloader_iterator = cycle(iter(train_dataloader))
    valid_dataloader_iterator = cycle(iter(valid_dataloader))

    model = FConvNetModel1()

    log = None
    if log_dir is not None:
        from tensorboardX import SummaryWriter
        # from dl.core.utils import SummaryWriter
        log = SummaryWriter(log_dir)

    optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss(weight=torch.from_numpy(train_class_loss_weights).float())

    for t in range(max_iter):
        train_batch = next(train_dataloader_iterator)
        batch_inputs = train_batch['inp_image']
        batch_labels = train_batch['lbl_image'].long()

        model.train()

        # zero the gradients (part of pytorch backprop)
        optimizer.zero_grad()

        # Compute the model output and loss (view flattens the input)
        model_outputs = model(batch_inputs)
        model_pred = torch.argmax(model_outputs, dim=1)
        t_loss_val = loss(model_outputs, batch_labels)
        t_acc_val = iou(model_pred, batch_labels)

        # Compute the gradient
        t_loss_val.backward()

        # Update the weights
        optimizer.step()

        if t % 10 == 0:
            model.eval()

            valid_batch = next(valid_dataloader_iterator)
            batch_inputs = valid_batch['inp_image']
            batch_labels = valid_batch['lbl_image'].long()

            model_outputs = model(batch_inputs)
            model_pred = torch.argmax(model_outputs, dim=1)
            v_acc_val = iou(model_pred, batch_labels)

            print(f'[{t:5d}] loss = {t_loss_val:1.6f} t_iou = {t_acc_val:1.6f} v_iou = {v_acc_val:1.6f}')

            if log is not None:
                log.add_scalar('train/loss', t_loss_val, t)
                log.add_scalar('train/iou', t_acc_val, t)
                log.add_scalar('val/iou', v_acc_val, t)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'model_state', 'fconvnet1.th'))


matplotlib.use('TkAgg')
# matplotlib.use('Agg')

COLORS = np.array([
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 255, 255)], dtype=np.uint8)


def test(batch_size=5):
    valid_dataloader = load_tux(path.join('..', 'data', 'fc_trainval', 'valid'), num_workers=4,
                            batch_size=batch_size)
    valid_dataloader_iterator = iter(valid_dataloader)

    fig, axes = plt.subplots(batch_size, 4, figsize=(4 * 2, batch_size * 2))
    valid_batch = next(valid_dataloader_iterator)
    batch_inputs = valid_batch['inp_image']
    batch_labels = valid_batch['lbl_image'].long()

    model = FConvNetModel1()
    model.load_state_dict(torch.load(os.path.join(dirname, 'model_state', 'fconvnet1.th'), map_location=lambda storage, loc: storage))
    model.eval()

    pred = model(batch_inputs).argmax(dim=1)

    for i, (image, label, p) in enumerate(zip(batch_inputs, batch_labels, pred)):
        image = np.transpose(image.detach().numpy(), [1, 2, 0])
        axes[i, 0].imshow(np.clip(image * [0.246, 0.286, 0.362] + [0.354, 0.488, 0.564], 0, 1))
        axes[i, 1].imshow(COLORS[label])
        axes[i, 2].imshow(COLORS[p])
        axes[i, 3].imshow(label == p)

    # Compute the confusion matrix
    plt.figure()
    CMs = []
    for it in range(50):
        valid_batch = next(valid_dataloader_iterator)
        batch_inputs = valid_batch['inp_image']
        batch_labels = valid_batch['lbl_image'].long()

        model_outputs = model(batch_inputs)
        model_pred = torch.argmax(model_outputs, dim=1)
        CMs.append(CM(model_pred, batch_labels).numpy())

    cm = np.mean(CMs, axis=0)
    cm = cm / np.max(cm)
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'IoU = {float(J(torch.as_tensor(cm)))}')
    for j in range(cm.shape[0]):
        for i in range(cm.shape[1]):
            plt.text(i, j, '%0.3f' % cm[j, i], horizontalalignment="center", color="black")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train',action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-i_tr', '--max_iter_tr', type=int, default=10000)
    parser.add_argument('-i_te', '--batch_size', type=int, default=5)
    parser.add_argument('-l', '--log_dir')
    args = parser.parse_args()
    if args.train:
        print('Start training')
        train(args.max_iter_tr, log_dir=args.log_dir)
        print('Training finished')
    if args.test:
        print('Testing')
        test(args.batch_size)
