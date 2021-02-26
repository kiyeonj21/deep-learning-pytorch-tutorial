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


import matplotlib.pyplot as plt

dirname = os.path.dirname(os.path.abspath(__file__))

downsample = nn.AvgPool2d(5, 4, 2)


def get_rgb_tr(batch_images):
    channel_1 = (batch_images[:, 0, :, :] * 0.246 + 0.354) * 255
    channel_2 = (batch_images[:, 1, :, :] * 0.286 + 0.488) * 255
    channel_3 = (batch_images[:, 2, :, :] * 0.362 + 0.564) * 255

    return torch.clamp(torch.stack((channel_1, channel_2, channel_3), 1).long().float(), 0, 255)


def train(max_iter, batch_size=64, log_dir=None):
    train_dataloader = load_tux(path.join('..', 'data', 'fc_trainval', 'train'), num_workers=4, crop=64)
    valid_dataloader = load_tux(path.join('..', 'data', 'fc_trainval', 'valid'), num_workers=4)

    train_dataloader_iterator = cycle(iter(train_dataloader))
    valid_dataloader_iterator = cycle(iter(valid_dataloader))

    model = FConvNetModel2()
    print(f"Num of parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

    log = None
    if log_dir is not None:
        from tensorboardX import SummaryWriter
        # from dl.core.utils import SummaryWriter
        log = SummaryWriter(log_dir)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    loss = nn.L1Loss(reduce=True)

    for t in range(max_iter):
        train_batch = next(train_dataloader_iterator)
        batch_targets = train_batch['inp_image']
        batch_labels = train_batch['lbl_image'].long()

        # .detach() is used to avoid backpropagation to the original high resolution image
        batch_inputs = downsample(batch_targets).detach()
        model.train()

        # zero the gradients (part of pytorch backprop)
        optimizer.zero_grad()

        # Compute the model output and loss (view flattens the input)
        model_outputs = model(batch_inputs, batch_labels)
        t_loss_val = loss(model_outputs, batch_targets)

        t_rgb_loss_val = loss(get_rgb_tr(model_outputs), get_rgb_tr(batch_targets))

        # Compute the gradient
        t_loss_val.backward()

        # Update the weights
        optimizer.step()

        if t % 10 == 0:
            model.eval()

            valid_batch = next(valid_dataloader_iterator)
            batch_targets = valid_batch['inp_image']
            batch_labels = valid_batch['lbl_image'].long()
            batch_inputs = downsample(batch_targets).detach()

            model_outputs = model(batch_inputs, batch_labels)
            v_rgb_loss_val = loss(get_rgb_tr(model_outputs), get_rgb_tr(batch_targets))

            print(f'[{t:5d}] t_loss = {t_loss_val:1.6f} '
                  f't_rgb_loss = {t_rgb_loss_val:1.6f} v_rgb_loss = {v_rgb_loss_val:1.6f}')

            if log is not None:
                log.add_scalar('train/loss', t_loss_val, t)
                # log.add_scalar('val/loss', v_loss_val, t)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'model_state', 'fconvnet2.th'))

COLORS = np.array([
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 255, 255)], dtype=np.uint8)


def get_rgb_te(batch_images):
    channel_1 = (batch_images[:, 0, :, :] * 0.246 + 0.354)
    channel_2 = (batch_images[:, 1, :, :] * 0.286 + 0.488)
    channel_3 = (batch_images[:, 2, :, :] * 0.362 + 0.564)

    return torch.clamp(torch.stack((channel_1, channel_2, channel_3), 1), 0, 1)


def test(batch_size=5):
    # valid_dataloader = load('valid', num_workers=4, batch_size=batch_size)
    valid_dataloader = load_tux(path.join('..', 'data', 'fc_trainval', 'valid'), num_workers=4,
                            batch_size=batch_size)
    valid_dataloader_iterator = iter(valid_dataloader)

    fig, axes = plt.subplots(batch_size, 3, figsize=(batch_size *2 * 3, batch_size*2))
    plt.subplots_adjust(hspace=2)
    valid_batch = next(valid_dataloader_iterator)
    batch_targets = valid_batch['inp_image']
    batch_labels = valid_batch['lbl_image'].long()
    batch_inputs = downsample(batch_targets)
    loss = nn.L1Loss()

    model = FConvNetModel2()
    model.load_state_dict(torch.load(os.path.join(dirname, 'model_state', 'fconvnet2.th'), map_location=lambda storage, loc: storage))
    model = model
    model.eval()

    pred = get_rgb_te(model(batch_inputs, batch_labels).detach())
    batch_targets = get_rgb_te(batch_targets.detach())
    batch_inputs = get_rgb_te(batch_inputs.detach())

    l1_loss = loss(pred, batch_targets)
    print(f'L1 Loss = {float(l1_loss.item() * 255)}')

    for i, (image, out_image, in_image) in enumerate(zip(batch_targets, batch_inputs, pred)):
        image = np.transpose(image.cpu().numpy(), [1, 2, 0])
        in_image = np.transpose(in_image.cpu().numpy(), [1, 2, 0])
        out_image = np.transpose(out_image.cpu().numpy(), [1, 2, 0])
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].imshow(image)
        axes[i, 1].set_title('Model Output')
        axes[i, 1].imshow(in_image)
        axes[i, 2].set_title('Input')
        axes[i, 2].imshow(out_image)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train',action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-i_tr', '--max_iter', type=int, default=10000)
    parser.add_argument('-i_te', '--batch_size', type=int, default=5)
    parser.add_argument('-l', '--log_dir')
    args = parser.parse_args()
    if args.train:
        print('Start training')
        train(args.max_iter, log_dir=args.log_dir)
        print('Training finished')
    if args.test:
        print('Testing')
        test(args.batch_size)

