import argparse, pickle
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn, optim
from dl.utils import load
from dl.models import *
from torchvision import transforms

dirname = os.path.dirname(os.path.abspath(__file__))

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.429, 0.505, 0.517], std=[0.274, 0.283, 0.347])
])


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(outputs)
    return outputs_idx.eq(labels.float()).float().mean()


all_transforms = transforms.Compose([
    transforms.ToPILImage(),

    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.429, 0.505, 0.517], std=[0.274, 0.283, 0.347])
])


def augment_data(inputs):
    return torch.stack([all_transforms(inp_i) for inp_i in inputs])


def transform_val(val_inputs):
    return torch.stack([val_transform(inp_i) for inp_i in val_inputs])


def train(iterations, batch_size=64, log_dir=None):
    train_inputs, train_labels = load(os.path.join('..', 'data', 'tux', 'tux_train.dat'))
    val_inputs, val_labels = load(os.path.join('..', 'data', 'tux', 'tux_valid.dat'))
    model = ConvNetModel2()
    log = None
    if log_dir is not None:
        from tensorboardX import SummaryWriter
        # from dl.dl.utils import SummaryWriter
        log = SummaryWriter(log_dir)

    # Enable L2 regularization of weights below
    optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()

    for iteration in range(iterations):
        model.train()
        # Construct a mini-batch
        batch = np.random.choice(train_inputs.shape[0], batch_size)

        # Perform data augmentation. (only needed during training)
        batch_inputs = augment_data(train_inputs[batch])
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
            batch = np.random.choice(val_inputs.shape[0], 256)
            batch_inputs = transform_val(val_inputs[batch])
            batch_labels = torch.as_tensor(val_labels[batch], dtype=torch.long)
            model_outputs = model(batch_inputs)
            v_acc_val = accuracy(model_outputs, batch_labels)

            print(f'[{iteration: 5d}] loss = {t_loss_val:1.6f} t_acc = {t_acc_val:1.6f} v_acc = {v_acc_val:1.6f}')

            if log is not None:
                log.add_scalar('train/loss', t_loss_val, iteration)
                log.add_scalar('train/acc', t_acc_val, iteration)
                log.add_scalar('val/acc', v_acc_val, iteration)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'model_state', 'convnet2.th'))  # Do NOT modify this line


def test(iterations, batch_size=256):
    # train_inputs, train_labels = load(os.path.join('tux_valid.dat'))
    train_inputs, train_labels = load(os.path.join('..', 'data', 'tux', 'tux_valid.dat'))

    model = ConvNetModel2()
    pred = lambda x: np.argmax(x.detach().numpy(), axis=1)

    model.load_state_dict(torch.load(os.path.join(dirname, 'model_state', 'convnet2.th')))
    model.eval()

    accuracies = []
    for iteration in range(iterations):
        # Construct a mini-batch
        batch = np.random.choice(train_inputs.shape[0], batch_size)
        batch_inputs = transform_val(train_inputs[batch])
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