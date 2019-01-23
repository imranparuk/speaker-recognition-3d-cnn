from __future__ import absolute_import

import functools
import numpy as np
import torch

from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import CrossEntropyLoss

from src.pytorch.dataset import VoxCelebDataset
from src.pytorch.model import _3d_cnn

# todo: fix this, find torch alternative
from tensorflow.python.keras.utils import to_categorical

def one_hot(a, num_classes):
    a = np.array(a)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def return_format_torch(_dict, _mapping, num_classes):
    ret_ = list()
    for speaker, files in _dict.items():
        for file in files:
            spk = _mapping.index(speaker)
            feat = file
            ret_.append([feat, spk])

    return map(list, zip(*ret_))

# https://github.com/pytorch/examples/blob/master/mnist/main.py

def train(model, device, train_loader, optimizer, epoch, log_interval=1):
    model.to(device)
    model.train()

    criterion = CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        print(target)
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target.long())
        loss.backward()

        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += CrossEntropyLoss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":

    initial_epoch = 0
    epochs = 1
    seed = 1
    input_shape = (1, 40, 80, 20)
    batch_size = 15
    validation_split = 0.1
    shuffle_dataset = True
    learning_rate = 0.001
    num_classes = 10
    path = "dummy_dir"
    info = [20, 80, 40]
    kwargs = {'num_workers': 1, 'pin_memory': True}


    dataset = VoxCelebDataset(path, batch_size, num_classes, info, return_format_torch)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    model = _3d_cnn(input_shape, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7, amsgrad=True)

    device = torch.device('cuda')

    for epoch in range(initial_epoch, epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

