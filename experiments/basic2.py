import argparse, pickle, os
import torch.nn as nn
from torch import tensor, save
import torch.optim as optim
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dl.models import LinearModel, DeepModel
import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def data_loader(batch=16):
    while True:
        inputs = np.random.uniform(size=[batch, 2])
        labels = np.zeros((batch, 2))
        idx = (np.linalg.norm(inputs, axis=1, ord=2) >= 1).astype(np.uint8)
        labels[np.arange(batch), idx] = 1
        yield tensor(inputs).float(), tensor(labels).float()


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(outputs)
    labels_idx = labels.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels_idx).float().mean()


def train_linear(model):
    log_step = 1000
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    running_loss, running_acc = 0., 0.
    for i, (inputs, labels) in zip(range(10000), data_loader()):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc.item()
        if i % log_step == 0:
            print(f'step {i:4d}, loss:{running_loss / log_step:1.6f}, acc: {running_acc / log_step:1.6f}')
            running_loss = 0.
            running_acc = 0.

    # Save the trained model
    dirname = os.path.dirname(os.path.abspath(__file__))
    torch.save(model.state_dict(), os.path.join(dirname, 'model_state', 'linear.th'))


def train_deep(model):
    log_step = 1000
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    running_loss, running_acc = 0., 0.
    for i, (inputs, labels) in zip(range(10000), data_loader()):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc.item()
        if i % log_step == 0:
            print(f'step {i:4d}, loss:{running_loss / log_step:1.6f}, acc: {running_acc / log_step:1.6f}')
            running_loss = 0.
            running_acc = 0.

    # Save the trained model
    dirname = os.path.dirname(os.path.abspath(__file__))
    torch.save(model.state_dict(), os.path.join(dirname, 'model_state', 'deep.th'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action="store_true")
    parser.add_argument('-test', action="store_true")
    parser.add_argument('-model', choices=['linear', 'deep'])
    args = parser.parse_args()

    if args.train:
        if args.model == 'linear':
            print('Start training linear model')
            train_linear(LinearModel())
        elif args.model == 'deep':
            print('Start training linear model')
            train_deep(DeepModel())
        print('Training finished')

    if args.test:
        linear_model = LinearModel()
        deep_model = DeepModel()

        dirname = os.path.dirname(os.path.abspath(__file__))

        f, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
        X, Y = np.meshgrid(np.arange(0, 1, 0.02), np.arange(0, 1, 0.02))
        inputs = np.array([X.flatten(), Y.flatten()]).T

        try:
            checkpoint = torch.load(os.path.join(dirname, 'model_state', 'linear.th'))
            linear_model.load_state_dict(checkpoint)
            outputs = linear_model(torch.tensor(inputs, dtype=torch.float)).detach().numpy()
            pos = inputs[outputs[:, 0] > 0]
            neg = inputs[outputs[:, 0] < 0]
            ax1.scatter(pos[:, 0], pos[:, 1], s=5)
            ax1.scatter(neg[:, 0], neg[:, 1], s=5)
            ax1.set_title('Linear Model')

        except FileNotFoundError:
            print("Could not find checkpoint, please make sure you train your linear model first")

        try:
            checkpoint = torch.load(os.path.join(dirname, 'model_state', 'deep.th'))
            deep_model.load_state_dict(checkpoint)
            outputs = deep_model(torch.tensor(inputs, dtype=torch.float)).detach().numpy()
            pos = inputs[outputs[:, 0] > 0]
            neg = inputs[outputs[:, 0] < 0]
            ax2.scatter(pos[:, 0], pos[:, 1], s=5)
            ax2.scatter(neg[:, 0], neg[:, 1], s=5)
            ax2.set_title('Deep Model')
        except FileNotFoundError:
            print("Could not find checkpoint, please make sure you train your linear model first")

        plt.show()
