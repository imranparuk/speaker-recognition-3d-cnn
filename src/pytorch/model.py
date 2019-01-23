# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init

class _3d_cnn(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        :param input_shape: input image shape, (h, w, c)
        """
        super(_3d_cnn, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1,  16, (5, 1, 3), stride=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(16, 16, (1, 9, 3), stride=(1, 2, 1)),
            nn.PReLU(),
            nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1)),

            nn.Conv3d(16, 32, kernel_size=(4, 1, 3), stride=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(32, 32, kernel_size=(1, 8, 3), stride=(1, 2, 1)),
            nn.PReLU(),
            nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1)),

            nn.Conv3d(32, 64, kernel_size=(3, 1, 3), stride=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=(1, 7, 3), stride=(1, 1, 1)),
            nn.PReLU(),

            nn.Conv3d(64, 128, kernel_size=(3, 1, 3), stride=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(128, 128, kernel_size=(1, 7, 3), stride=(1, 1, 1)),
            nn.PReLU(),
        )

        # Compute number of input features for the last fully-connected layer
        input_shape = (1,) + input_shape
        # x = Variable(torch.rand(input_shape), requires_grad=False)
        # x = self.features(x)
        # x = Flatten()(x)
        # n = x.size()[1]

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(4608, num_classes)
            # nn.Softmax(dim=1)
        )

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            weight_init_helper(m)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def weight_init_helper(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
        https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


