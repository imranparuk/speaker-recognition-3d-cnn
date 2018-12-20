from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.utils import to_categorical

from src.common.helper import VoxCelebHelper
from src.common.helper import preproc

def return_format_keras(_dict, _mapping, num_classes):
    ret_ = list()
    for speaker, files in _dict.items():
        for file in files:
            spk = _mapping.index(speaker)
            feat = file
            ret_.append([feat, to_categorical(spk, num_classes)])

    return map(list, zip(*ret_))


class DataGenerator(Sequence):

    def __init__(self, path, num_classes=10, batch_size=20, info=[20, 80, 40], ret_func=return_format_keras):

        self.features, self.labels = VoxCelebHelper(path, num_classes, info, ret_func)()
        self.data_len = len(self.features)

        self.info = info

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

        self.batch_size = batch_size

        if self.batch_size > self.data_len:
            self.batch_size = self.data_len

        self.shuffle = True

        self.list_IDs = range(0, self.data_len)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        idxs = [self.list_IDs[k] for k in indexes]
        ret_feat = []
        ret_labs = []

        for idx in idxs:
            f, l = self.__data_generation(idx)
            ret_feat.append(f)
            ret_labs.append(l)

        np_f = np.vstack(ret_feat)
        np_l = np.vstack(ret_labs)

        return np_f, np_l

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idx):
        feat = preproc(str(self.features[idx]), self.info)
        feat = np.expand_dims(feat, axis=0)

        lab = self.labels[idx]

        return feat, lab

