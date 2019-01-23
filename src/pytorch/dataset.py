import numpy as np
from torch.utils.data import Dataset

from src.common.helper import VoxCelebHelper
from src.common.helper import preproc

class VoxCelebDataset(Dataset):
    def __init__(self, path, batch_size=15, num_classes=2, info=[20,80,40], ret_func=None):

        self.num_utterances = 20
        self.num_frames = 80
        self.num_coefficient = 40

        self.features, self.labels = VoxCelebHelper(path, num_classes, info, ret_func)()
        self.data_len = len(self.features)

        self.info = info

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)


        self.batch_size = batch_size


        self.info = info


    def __getitem__(self, index):

        input = self.features[index,...]
        labels = self.labels[index,...]

        feat = preproc(str(input), self.info).T

        return feat, labels

    def __len__(self):
        return self.data_len



