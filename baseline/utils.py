from __future__ import print_function

import warnings

import numpy as np
import pandas as pd
import soundfile
import os
import librosa
import torch
from torch import nn


class ManyHotEncoder:
    """ Encode labels into numpy arrays where 1 correspond to presence of the class and 0 absence.
        Multiple 1 can appear on the same line, it is for multi label problem.
    Args:
        labels: list, the classes which will be encoded
        n_frames: int, (Default value = None) only useful for strong labels. The number of frames of a segment.
    Attributes:
        labels: list, the classes which will be encoded
        n_frames: int, only useful for strong labels. The number of frames of a segment.
    """
    def __init__(self, labels, n_frames=None):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.n_frames = n_frames

    def encode_weak(self, labels):
        """ Encode a list of weak labels into a numpy array

        Args:
            labels: list, list of labels to encode (to a vector of 0 and 1)

        Returns:
            numpy.array
            A vector containing 1 for each label, and 0 everywhere else
        """
        # useful for tensor empty labels
        if type(labels) is str:
            if labels == "empty":
                y = np.zeros(len(self.labels)) - 1
                return y
        y = np.zeros(len(self.labels))
        for label in labels:
            i = self.labels.index(label)
            y[i] = 1
        return y

    def encode_strong_df(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        assert self.n_frames is not None, "n_frames need to be specified when using strong encoder"
        if type(label_df) is str:
            if label_df == 'empty':
                y = np.zeros((self.n_frames, len(self.labels))) - 1
                return y
        y = np.zeros((self.n_frames, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    if not pd.isna(row["event_label"]):
                        i = self.labels.index(row["event_label"])
                        onset = int(row["onset"])
                        offset = int(row["offset"])
                        y[onset:offset, i] = 1  # means offset not included (hypothesis of overlapping frames, so ok)

        elif type(label_df) in [pd.Series, list, np.ndarray]:  # list of list or list of strings
            if type(label_df) is pd.Series:
                if {"onset", "offset", "event_label"}.issubset(label_df.index):  # means only one value
                    if not pd.isna(label_df["event_label"]):
                        i = self.labels.index(label_df["event_label"])
                        onset = int(label_df["onset"])
                        offset = int(label_df["offset"])
                        y[onset:offset, i] = 1
                    return y

            for event_label in label_df:
                # List of string, so weak labels to be encoded in strong
                if type(event_label) is str:
                    if event_label is not "":
                        i = self.labels.index(event_label)
                        y[:, i] = 1

                # List of list, with [label, onset, offset]
                elif len(event_label) == 3:
                    if event_label[0] is not "":
                        i = self.labels.index(event_label[0])
                        onset = int(event_label[1])
                        offset = int(event_label[2])
                        y[onset:offset, i] = 1

                else:
                    raise NotImplementedError("cannot encode strong, type mismatch: {}".format(type(event_label)))

        else:
            raise NotImplementedError("To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                                      "columns, or it is a list or pandas Series of event labels, "
                                      "type given: {}".format(type(label_df)))
        return y

    def decode_weak(self, labels):
        """ Decode the encoded weak labels
        Args:
            labels: numpy.array, the encoded labels to be decoded

        Returns:
            list
            Decoded labels, list of string

        """
        result_labels = []
        for i, value in enumerate(labels):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def decode_strong(self, labels):
        """ Decode the encoded strong labels
        Args:
            labels: numpy.array, the encoded labels to be decoded
        Returns:
            list
            Decoded labels, list of list: [[label, onset offset], ...]

        """
        result_labels = []
        for i, label_column in enumerate(labels.T):
            # Find the changes in the activity_array
            change_indices = np.logical_xor(label_column[1:], label_column[:-1]).nonzero()[0]

            # Shift change_index with one, focus on frame after the change.
            change_indices += 1

            if label_column[0]:
                # If the first element of activity_array is True add 0 at the beginning
                change_indices = np.r_[0, change_indices]

            if label_column[-1]:
                # If the last element of activity_array is True, the stop indices is the length of segment
                change_indices = np.r_[change_indices, label_column.size]

            # Reshape the result into two columns
            change_indices = change_indices.reshape((-1, 2))

            # append [label, onset, offset] in the result list
            for row in change_indices:
                result_labels.append([self.labels[i], row[0], row[1]])
        return result_labels


def read_audio(path, target_fs=None):
    """ Read a wav file
    Args:
        path: str, path of the audio file
        target_fs: int, (Default value = None) sampling rate of the returned audio file, if not specified, the sampling
            rate of the audio file is taken

    Returns:
        tuple
        (numpy.array, sampling rate), array containing the audio at the sampling rate given

    """
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def write_audio(path, audio, sample_rate):
    """ Save an audio file
    Args:
        path: str, path used to store the audio
        audio: numpy.array, audio data to store
        sample_rate: int, the sampling rate
    """
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


def create_folder(fd):
    """ Create folders of a path if not exists
    Args:
        fd: str, path to the folder to create
    """
    if not os.path.exists(fd):
        os.makedirs(fd)


def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    length = len(x)
    shape = x.shape
    if length < max_len:
        pad_shape = (max_len - length,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    elif length > max_len:
        x_new = x[0:max_len]
    else:
        x_new = x
    return x_new


def weights_init(m):
    """ Initialize the weights of some layers of neural networks, here Conv2D, BatchNorm, GRU, Linear
        Based on the work of Xavier Glorot
    Args:
        m: the model to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('GRU') != -1:
        for weight in m.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def to_cuda_if_available(list_args):
    """ Transfer object (Module, Tensor) to GPU if GPU available
    Args:
        list_args: list, list of objects to put on cuda if available

    Returns:
        list
        Objects on GPU if GPUs available
    """
    if torch.cuda.is_available():
        for i in range(len(list_args)):
            list_args[i] = list_args[i].cuda()
    return list_args


def to_cpu(list_args):
    """ Transfer object (Module, Tensor) to CPU if GPU available
        Args:
            list_args: list, list of objects to put on CPU (if not already)

        Returns:
            list
            Objects on CPU
        """
    if torch.cuda.is_available():
        for i in range(len(list_args)):
            list_args[i] = list_args[i].cpu()
    return list_args


class SaveBest:
    """ Callback of a model to store the best model based on a criterion
    Args:
        model: torch.nn.Module, the model which will be tracked
        val_comp: str, (Default value = "inf") "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
    Attributes:
        model: torch.nn.Module, the model which will be tracked
        val_comp: str, "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
        best_val: float, the best values of the model based on the criterion chosen
        best_epoch: int, the epoch when the model was the best
        current_epoch: int, the current epoch of the model
    """
    def __init__(self, val_comp="inf"):
        self.comp = val_comp
        if val_comp == "inf":
            self.best_val = np.inf
        elif val_comp == "sup":
            self.best_val = 0
        else:
            raise NotImplementedError("value comparison is only 'inf' or 'sup'")
        self.best_epoch = 0
        self.current_epoch = 0

    def apply(self, value):
        """ Apply the callback
        Args:
            value: float, the value of the metric followed
            model_path: str, the path where to store the model
            parameters: dict, the parameters to be saved by pytorch in the file model_path.
            If model_path is not None, parameters is not None, and the other way around.
        """
        decision = False
        if self.current_epoch == 0:
            decision = True
        if (self.comp == "inf" and value < self.best_val) or (self.comp == "sup" and value > self.best_val):
            self.best_epoch = self.current_epoch
            self.best_val = value
            decision = True
        self.current_epoch += 1
        return decision


class EarlyStopping:
    """ Callback of a model to store the best model based on a criterion
    Args:
        model: torch.nn.Module, the model which will be tracked
        patience: int, number of epochs with no improvement before stopping the model
        val_comp: str, (Default value = "inf") "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
    Attributes:
        model: torch.nn.Module, the model which will be tracked
        patience: int, number of epochs with no improvement before stopping the model
        val_comp: str, "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
        best_val: float, the best values of the model based on the criterion chosen
        best_epoch: int, the epoch when the model was the best
        current_epoch: int, the current epoch of the model
    """
    def __init__(self, model, patience, val_comp="inf"):
        self.model = model
        self.patience = patience
        self.val_comp = val_comp
        if val_comp == "inf":
            self.best_val = np.inf
        elif val_comp == "sup":
            self.best_val = 0
        else:
            raise NotImplementedError("value comparison is only 'inf' or 'sup'")
        self.current_epoch = 0
        self.best_epoch = 0

    def apply(self, value):
        """ Apply the callback

        Args:
            value: the value of the metric followed
        """
        current = False
        if self.val_comp == "inf":
            if value < self.best_val:
                current = True
        if self.val_comp == "sup":
            if value > self.best_val:
                current = True
        if current:
            self.best_val = value
            self.best_epoch = self.current_epoch
        elif self.current_epoch - self.best_epoch > self.patience:
            return True
        self.current_epoch += 1
        return False


def change_view_frames(array, nb_frames):
    array = array.view(-1, array.shape[1], nb_frames, array.shape[-1])
    return array


def save_model(state, filename=None, overwrite=True):
    """ Function to save Pytorch models.
    # Argument
        dic_params: Dict. Must includes "model" (and possibly "optimizer")which is a dict with "name","args","kwargs",
        example:
        state = {
                 'epoch': epoch_ + 1,
                 'model': {"name": classif_model.get_name(),
                           'args': rnn_args,
                           "kwargs": rnn_kwargs,
                           'state_dict': classif_model.state_dict()},
                 'optimizer': {"name": classif_model.get_name(),
                               'args': '',
                               "kwargs": optim_kwargs,
                               'state_dict': optimizer_classif.state_dict()},
                 'loss': loss_mean_bce
                 }
        filename: String. Where to save the model.
        overwrite: Bool. Whether to overwrite existing file or not.
    # Raises
        Warning if filename exists and overwrite isn't set to True.
    """
    if os.path.isfile(filename):
        if overwrite:
            os.remove(filename)
            torch.save(state, filename)
        else:
            warnings.warn('Found existing file at {}'.format(filename) +
                          'specify `overwrite=True` if you want to overwrite it')
    else:
        torch.save(state, filename)


# ##################
# MANDATORY FOR get_class to work !!!!!!
# It puts the name of the class in globals()
# ###################
# from models.CNN import CNN
# from models.CRNN import CRNN
# from torch.optim import Adam, Adadelta
# def get_class(name):
#     try:
#         cls = globals()[name]
#     except KeyError as ke:
#         raise KeyError("Impossible to load the object from a string, check the class name and "
#                        "if the situation is covered")
#     return cls


# def load_model(filename, ema=False, return_optimizer=False, return_state=False):
#     """ Function to load Pytorch models.
#     # Argument
#         filename: String. Where to load the model from.
#         example:
#         state = {
#                  'epoch': epoch_ + 1,
#                  'model': {"name": classif_model.get_name(),
#                            'args': rnn_args,
#                            "kwargs": rnn_kwargs,
#                            'state_dict': classif_model.state_dict()},
#                  'optimizer': {"name": classif_model.get_name(),
#                                'args': '',
#                                "kwargs": optim_kwargs,
#                                'state_dict': optimizer_classif.state_dict()},
#                  'loss': loss_mean_bce
#                  }
#         return_optimizer: Bool. Whether to return optimizer or not.
#     # Returns
#         A Pytorch model instance.
#         An Pytorch optimizer instance.
#     """
#     state = torch.load(filename)
#
#     def get_model(model_key):
#         model = get_class(state[model_key]["name"])(*state[model_key]['args'],
#                                                     **state[model_key]['kwargs'])
#         # Assuming we load model
#         model.load(parameters=state[model_key['state_dict']])
#         return model
#
#     model = get_model("model")
#     to_return = (model,)
#     if ema:
#         model_ema = get_model("model_ema")
#
#     if return_optimizer:
#         optimizer = get_class(state["optimizer"]["name"])(filter(lambda p: p.requires_grad, model.parameters()),
#                                                           *state['optimizer']['args'],
#                                                           **state['optimizer']['kwargs'])
#         optimizer.load_state_dict(state['optimizer']['state_dict'])
#         if return_state:
#             return model, optimizer, state
#         return model, optimizer
#     else:
#         if return_state:
#             return model, state
#         return model
#

class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.avg:{format}}".format(self=self, format=format)