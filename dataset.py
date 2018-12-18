from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
from scipy.io import wavfile
from pathlib import Path

from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.utils import to_categorical

import speechpy as speechpy

class DataGenerator(Sequence):

    def __init__(self, path, num_classes=10, batch_size=20, num_utterances=20, num_frames=80, num_coefficient=40):

        self.num_classes = num_classes

        self.check_dataset_dict = self.generate_dataset_dict_helper(path)
        self.dataset_dict = self.check_dataset(self.check_dataset_dict)

        self.mapping = self.create_speaker_id_list(self.dataset_dict)

        features, labels = self.return_format(self.dataset_dict)

        self.features = np.array(features)
        self.labels = np.array(labels)

        self.data_len = len(features)
        self.batch_size = batch_size

        if self.batch_size > self.data_len:
            self.batch_size = self.data_len

        self.shuffle = True

        self.num_utterances = num_utterances
        self.num_frames = num_frames
        self.num_coefficient = num_coefficient

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
        feat = self.preproc(self.features[idx])
        lab = self.labels[idx]

        return feat, lab

    def preproc(self, file):
        sample_freq, raw_seq = wavfile.read(str(file))

        raw_seq = raw_seq.astype(np.float32)
        logenergy = speechpy.lmfe(raw_seq, sampling_frequency=sample_freq, frame_length=0.025, frame_stride=0.01,
                                          num_filters=self.num_coefficient, fft_length=512, low_frequency=0,
                                          high_frequency=None)

        feature_cube = np.ones((self.num_utterances, self.num_frames, self.num_coefficient, 1), dtype=np.float32)

        idx = range(self.num_utterances)
        for num, index in enumerate(idx):
            log_energy_slice = logenergy[index:index + self.num_frames, :]
            feature_cube[num, :log_energy_slice.shape[0], :log_energy_slice.shape[1], 0] = log_energy_slice

        feature_cube = np.expand_dims(feature_cube, axis=0)

        return feature_cube

    @staticmethod
    def create_speaker_id_list(_speaker_dict):
        _ret_list = list(_speaker_dict.keys())
        return _ret_list

    @staticmethod
    def generate_dataset_dict_helper( dataset_path):
        ret_dict = {}
        for parent, dirs, files in os.walk(dataset_path):

            # top level dir
            if dataset_path is not parent:

                # there are no subdirs
                if not len(dirs) > 0:
                    path_i = Path(parent)
                    path_i_parent = str(path_i)
                    path_y_list = [str(Path(parent) / Path(x)) for x in files]

                    if path_i_parent in list(ret_dict.keys()):
                        ret_dict[str(path_i_parent)].extend(path_y_list)
                    else:
                        ret_dict[str(path_i_parent)] = list(path_y_list)
        return ret_dict

    def check_dataset(self, _dict):
        total_count = sum(len(v) for v in _dict.values())
        current_count = 0
        _checked_dict = dict()

        for speaker, data in _dict.items():
            temp_data_list = list()

            filename_w_ext = os.path.basename(speaker)
            filename, file_extension = os.path.splitext(filename_w_ext)

            for file in data:
                try:
                    with open(file, 'rb') as f:
                        riff_size, _ = wavfile._read_riff_chunk(f)
                        file_size = os.path.getsize(file)

                    # Assertion error.
                    assert riff_size == file_size and os.path.getsize(file) > 1000, "Bad file!"

                    temp_data_list.append(file)
                except OSError as err:
                    print('OS error: {0}'.format(err))

                except ValueError:
                    print('file {0} is corrupted!'.format(file))

                self.printProgressBar(current_count + 1, total_count, prefix='Progress (Checking items):',
                                      suffix='Complete', length=50)
                current_count += 1

            _checked_dict[filename] = temp_data_list
        return _checked_dict

    @staticmethod
    def convert_dict_to_list(_dict, labels_dict):
        return_list = []
        for label, feat_list in _dict.items():
            for feat in feat_list:
                return_list.append((feat, labels_dict[label]))
        return return_list

    @staticmethod
    def label_converter(_labels_list):
        _labels_dict = {}
        count = 0
        for label in _labels_list:
            if label not in _labels_dict.keys():
                _labels_dict[label] = count
                count += 1
        return _labels_dict

    def return_format(self, _dict):

        ret_ = list()
        for speaker, files in _dict.items():
            for file in files:
                spk = self.mapping.index(speaker)
                feat = file
                ret_.append([feat, to_categorical(spk, self.num_classes)])

        return map(list, zip(*ret_))

    @staticmethod
    def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        """
        Call in a loop to create terminal progress bar
        Stolen from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()
