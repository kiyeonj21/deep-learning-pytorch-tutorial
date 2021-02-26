import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 100)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        return self.fc2(self.rl1(self.fc1(x)))


class VectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 100)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 6)
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        return self.fc2(self.rl1(self.fc1(x)))


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2,2)

    def forward(self, x):
        x = self.linear(x)
        return x


class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        return x


class ConvNetModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, 1)
        self.fc = nn.Linear(6272, 6)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        '''
        Input: a series of N input images x. size (N, 64*64*3)
        Output: a prediction of each input image. size (N,6)
        '''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(-1, 6272)
        x = self.fc(x)

        return x


class ConvNetModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 1),
            nn.ReLU(True),
            # torch.nn.GroupNorm(4,32, affine=False),
            nn.Conv2d(32, 64, 5, 2, 1),
            nn.ReLU(True),
            # torch.nn.GroupNorm(4,64, affine=False),
            nn.Conv2d(64, 128, 5, 2, 1),
            nn.ReLU(True),
            # torch.nn.GroupNorm(4,128, affine=False),
            nn.Conv2d(128, 6, 5, 2, 1),
            nn.AvgPool2d(3)
        )

    def forward(self, x):
        '''
        Input: a series of N input images x. size (N, 64*64*3)
        Output: a prediction of each input image. size (N,6)
        '''

        return self.model(x).view(-1, 6)


class Block(nn.Module):
    # resnet block
    def __init__(self, in_out_channel, inner_dim, kernel_size=1, padding=0):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_out_channel),
            nn.Conv2d(in_out_channel, inner_dim, kernel_size, 1, padding),
            nn.ReLU(True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, inner_dim, kernel_size, 1, padding),
            nn.ReLU(True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, inner_dim, kernel_size, 1, padding),
            nn.ReLU(True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, in_out_channel, kernel_size, 1, padding),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 1),
            nn.ReLU(True),
            Block(32, 16, 3, 1),

            nn.Conv2d(32, 64, 5, 2, 1),
            nn.ReLU(True),
            Block(64, 32, 3, 1),

            nn.Conv2d(64, 128, 5, 2, 1),
            nn.ReLU(True),
            Block(128, 64, 3, 1),

            nn.Conv2d(128, 6, 5, 2, 1),
            nn.AvgPool2d(3)
        )

    def forward(self, x):
        '''
        Input: a series of N input images x. size (N, 64*64*3)
        Output: a prediction of each input image. size (N,6)
        '''
        x = self.model(x)

        return x.view(-1, 6)


class FConvNetModel1(nn.Module):
    def __init__(self, ks=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, ks, 2, ks // 2)
        self.conv2 = nn.Conv2d(16, 32, ks, 2, ks // 2)
        self.conv3 = nn.Conv2d(32, 64, ks, 2, ks // 2)
        self.conv4 = nn.Conv2d(64, 128, ks, 2, ks // 2)

        self.upconv1 = nn.ConvTranspose2d(128, 64, ks, 2, ks // 2, 1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, ks, 2, ks // 2, 1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, ks, 2, ks // 2, 1)
        self.upconv4 = nn.ConvTranspose2d(16, 6, ks, 2, ks // 2, 1)
        nn.init.constant_(self.upconv4.weight, 0)
        nn.init.constant_(self.upconv4.bias, 0)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        d1 = self.relu(self.conv1(x))
        d2 = self.relu(self.conv2(d1))
        d3 = self.relu(self.conv3(d2))
        d4 = self.relu(self.conv4(d3))

        u4 = self.relu(self.upconv1(d4))
        u3 = self.relu(self.upconv2(u4 + d3))
        u2 = self.relu(self.upconv3(u3 + d2))
        u1 = self.upconv4(u2 + d1)

        return u1


def one_hot(x, n=6):
    batch_size, h, w = x.size()
    x = (x.view(-1, h, w, 1) == torch.arange(n, dtype=x.dtype, device=x.device)[None]).float() - torch.as_tensor(
        [0.6609, 0.0045, 0.017, 0.0001, 0.0036, 0.314], dtype=torch.float, device=x.device)
    x = x.permute(0, 3, 1, 2)
    return x


class FConvNetModel2(nn.Module):
    def __init__(self, ks=5, channels=[16, 32]):

        super().__init__()
        n_input_cat = 9
        c0 = n_input_cat
        # Let's define some down-convolutions (with relu)
        self.conv = nn.ModuleList()
        for c in channels:
            self.conv.append(nn.Sequential(
                nn.BatchNorm2d(c0, affine=False),
                nn.Conv2d(c0, c, ks, 2, (ks - 1) // 2),
                nn.ReLU(True),
            ))
            c0 = c + n_input_cat

        # One convolution at the low res
        self.central_conv = nn.Sequential(nn.Conv2d(c0, channels[-1], ks, 1, (ks - 1) // 2),
                                          nn.ReLU(True))
        c0 = channels[-1]

        # And back up
        self.upconv = nn.ModuleList()
        for c in channels[::-1]:
            self.upconv.append(nn.Sequential(
                nn.BatchNorm2d(c0, affine=False),
                nn.ConvTranspose2d(c0, c, ks, 2, (ks - 1) // 2, 1),
                nn.ReLU(True),
            ))
            c0 = c

        # The pixels are generated by a single conv
        self.final_conv = nn.Conv2d(c0, 3, ks, 1, (ks - 1) // 2)
        nn.init.constant_(self.final_conv.weight, 0)

    def forward(self, lr_image, labels):

        # We'll feed the upsampled image into teach down-convolution
        hr_image = F.interpolate(lr_image, scale_factor=4, mode='bilinear', align_corners=False)
        mr_image = F.interpolate(lr_image, scale_factor=2, mode='bilinear', align_corners=False)

        # as well as the labels map at the appropriate resolution
        hr_labels = one_hot(labels)
        mr_labels = F.interpolate(hr_labels, scale_factor=0.5)
        lr_labels = F.interpolate(hr_labels, scale_factor=0.25)

        # Let's concat the input
        x0 = torch.cat((hr_image, hr_labels), 1)
        x1 = torch.cat((mr_image, mr_labels), 1)
        x2 = torch.cat((lr_image, lr_labels), 1)

        # and run the network
        d1 = self.conv[0](x0)
        d2 = self.conv[1](torch.cat((d1, x1), dim=1))
        d3 = self.central_conv(torch.cat((d2, x2), dim=1))

        u2 = self.upconv[0](d3)
        u1 = self.upconv[1](u2)
        delta = self.final_conv(u1)

        # Instead of outputting the image, we only produce the delta over linear upsampling
        return delta + hr_image
    # return image


class SeqPredictor:
    def __init__(self, model):
        self.model = model
        self.hist = []

    def __call__(self, input):
        """
        @param input: A single input of shape (6,) indicator values (float: 0 or 1)
        @return The logit of a binary distribution of output actions (6 floating point values between -infty .. infty)
        """
        self.hist.append(input)
        if len(self.hist) > self.model.width:
            self.hist = self.hist[-self.model.width:]
        x = torch.stack(self.hist, dim=-1)[None]
        return self.model(x)[0, :, -1]


class RNNModel1(nn.Module):
    def __init__(self):

        super().__init__()
        # The number of sentiment classes
        self.target_size = 6
        self.width = 100

        # The Dropout Layer Probability. Same for all layers
        self.dropout_prob = 0.0

        # Option to use a stacked LSTM
        self.num_lstm_layers = 1

        # Option to Use a bidirectional LSTM

        self.isBidirectional = False

        if self.isBidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # The Number of Hidden Dimensions in the LSTM Layers
        self.hidden_dim = 32

        self.setup_model_layers()

        self.init_weights()

    def setup_model_layers(self):
        """
        Function that creates the all the layers of the model
        """

        # Stacked LSTM with dropout
        self.lstm_layer = nn.LSTM(
            input_size=6,
            hidden_size=self.hidden_dim,
            num_layers=self.num_lstm_layers,
            bidirectional=self.isBidirectional,
            batch_first=True
            )

        # Create the Dense layer
        self.dense_layer = nn.Linear(
            in_features=self.num_directions * self.hidden_dim,
            out_features=self.target_size
            )

        # Use a dropout layer to prevent over-fitting of the model
        self.dropout_layer = nn.Dropout(self.dropout_prob)

    def init_weights(self):
        nn.init.constant_(self.dense_layer.bias, 0.0)
        nn.init.constant_(self.dense_layer.weight, 0.0)

    def forward(self, input_seq):
        """
        IMPORTANT: Do not change the function signature of the forward() function unless the grader won't work.
        @param input: A sequence of input actions (batch_size x 6 x sequence_length)
        @return The logit of a binary distribution of output actions (6 floating point values between -infty .. infty). Shape: batch_size x 6 x sequence_length
        """
        input_seq = input_seq.permute(0, 2, 1)
        lstm_output, hidden_state = self.lstm_layer(input_seq)

        dense_outputs = self.dropout_layer(
            self.dense_layer(lstm_output.contiguous().view(-1, self.num_directions * self.hidden_dim)))

        dense_outputs = dense_outputs.view(-1, input_seq.size(1), self.target_size)

        out = dense_outputs
        return out.permute(0, 2, 1)

    def predictor(self):
        return SeqPredictor(self)


class RNNModel2(nn.Module):
    # CNN-based RNNModel
    def __init__(self, channels=[16, 32, 16], ks=5):
        super().__init__()

        c0 = 6
        layers = []
        self.width = 1
        for c in channels:
            layers.append(nn.Conv1d(c0, c, ks))
            layers.append(nn.LeakyReLU())
            c0 = c
            self.width += ks - 1
        layers.append(nn.Conv1d(c0, 6, ks))
        self.width += ks - 1

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        """
        @param input: A sequence of input actions (batch_size x 6 x sequence_length)
        @return The logit of a binary distribution of output actions (6 floating point values between -infty .. infty). Shape: batch_size x 6 x sequence_length
        """
        # Let's pad only on the left
        x = F.pad(input, (self.width - 1, 0))
        return self.model(x)

    def predictor(self):
        return SeqPredictor(self)