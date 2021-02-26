import numpy as np
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from skimage import io
from os import path
import matplotlib.pyplot as plt

from itertools import cycle

def load(filename, W=64, H=64):
    """
    Loads the data that is provided
    @param filename: The name of the data file. Can be either 'tux_train.dat' or 'tux_val.dat'
    @return images: Numpy array of all images where the shape of each image will be W*H*3
    @return labels: Array of integer labels for each corresponding image in images
    """

    try:
        data = np.fromfile(filename, dtype=np.uint8).reshape((-1, W * H * 3 + 1))
    except Exception as e:
        print('Check if the filepath of the dataset is {}'.format(os.path(filename)))

    images, labels = data[:, :-1].reshape((-1, H, W, 3)), data[:, -1]
    return images, labels


class TuxDataset(Dataset):
    """
    Dataset class that reads the Tux dataset
    """

    def __init__(self, data_folder, crop=None):
        from os import path
        from glob import glob

        self.data_folder = data_folder

        # Load all data into memory
        print("[I] Loading data from %s" % data_folder)
        self.filenames = glob(path.join(data_folder, '*-img.png'))
        # self.filenames = glob(path.join(path.expanduser('~'),'Data','fc_trainval',data_folder,'*-img.png'))
        self.crop = crop
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.354, 0.488, 0.564], std=[0.246, 0.286, 0.362])
        ])

    def __len__(self):
        return len(self.filenames)

    def _mask(self, im):
        r = (im[:, :, 0] > 0).astype(np.uint8) + 2 * (im[:, :, 1] > 0).astype(np.uint8) + 4 * (im[:, :, 2] > 0).astype(
            np.uint8)
        r[r > 5] = 5
        return r

    def __getitem__(self, idx):
        I, L = io.imread(self.filenames[idx]), self._mask(io.imread(self.filenames[idx].replace('-img', '-lbl')))
        if self.crop is not None:
            y, x = np.random.choice(I.shape[0] - self.crop), np.random.choice(I.shape[1] - self.crop)
            I, L = I[y:self.crop + y, x:self.crop + x], L[y:self.crop + y, x:self.crop + x]
            if np.random.random() > 0.5:
                I, L = np.ascontiguousarray(I[:, ::-1]), np.ascontiguousarray(L[:, ::-1])

        ret = {
            'inp_image': self.image_transform(I),
            'lbl_image': torch.tensor(L)
        }
        return ret


def load_tux(data_folder, num_workers=0, batch_size=32, **kwargs):
    dataset = TuxDataset(data_folder, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


class ActionDataset(Dataset):
    """
    Dataset class that reads the Actions dataset
    """

    def __init__(self, data_filepath, crop=None):
        self.crop = crop

        # Load all data into memory
        print("[I] Loading data from %s" % data_filepath)
        with open(data_filepath, 'r') as f:
            self.action_seqs = [np.fromstring(l, dtype=np.uint8, sep=',') for l in f.readlines()]
        # Delete any short seqs
        if self.crop is not None:
            self.action_seqs = [s for s in self.action_seqs if len(s) > self.crop]

    def __len__(self):
        return len(self.action_seqs)

    def __getitem__(self, idx):
        r = self.action_seqs[idx]
        if self.crop is not None:
            s = np.random.choice(len(r) - self.crop + 1)
            r = r[s:s + self.crop]
        return np.unpackbits(r[None], axis=0)[:6]


def load_action(data_filepath, num_workers=0, batch_size=32, **kwargs):
    dataset = ActionDataset(data_filepath, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def show_image(batch_size=5, dataset = 'fc_trainval'):
    COLORS = np.array([
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (255, 255, 255)], dtype=np.uint8)

    if dataset =='fc_trainval':
        dataloader = load_tux(path.join('..', 'data', 'fc_trainval', 'valid'),
                              num_workers=0, batch_size=batch_size)
        dataloader_iterator = iter(dataloader)
        batch = next(dataloader_iterator)
        batch_inputs = batch['inp_image']
        batch_labels = batch['lbl_image'].long()
        plt.figure()
        fig, axes = plt.subplots(batch_size, 2, figsize=(4 * 2, batch_size * 2))
        for i, (image, label) in enumerate(zip(batch_inputs, batch_labels)):
            image = np.transpose(image.detach().numpy(), [1, 2, 0])
            # print(image.shape)
            axes[i, 0].imshow(np.clip(image * [0.246, 0.286, 0.362] + [0.354, 0.488, 0.564], 0, 1))
            axes[i, 1].imshow(COLORS[label])
        plt.show()

    if dataset =='tux':
        train_inputs, train_labels = load(os.path.join('..', 'data', 'tux', 'tux_train.dat'))
        batch = np.random.choice(train_inputs.shape[0], batch_size)
        batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)
        batch_labels = torch.as_tensor(train_labels[batch], dtype=torch.long)
        plt.figure()
        fig, axes = plt.subplots(batch_size, 1, figsize=(1 * 2, batch_size * 2))
        for i, (image, label) in enumerate(zip(batch_inputs, batch_labels)):
            image = np.uint8(image.numpy())
            axes[i].imshow(image)
            axes[i].set_title(f'label is {label.item()}')
        plt.show()

    if dataset =='action_trainval':
        data = ActionDataset(path.join('..', 'data', 'action_trainval', 'valid.dat'))
        batch = np.random.choice(len(data.action_seqs), batch_size)
        # batch_inputs = torch.as_tensor(data[batch], dtype=torch.float32)

        plt.figure()
        fig, axes = plt.subplots(batch_size, 1, figsize=(2 * 2, 2 * 2))
        for i, (idx) in enumerate(batch):
            image = data[i]
            axes[i].imshow(image[:,850:900])
        plt.show()