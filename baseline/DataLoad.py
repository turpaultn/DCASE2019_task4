import bisect
import itertools

import numpy as np
import pandas as pd
import torch
import random
import librosa
import warnings
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from utils import pad_trunc_seq
from config import LOG

torch.manual_seed(0)
random.seed(0)


class DataLoadDf(Dataset):
    """ Class derived from pytorch Dataset
    Prepare the data to be use in a batch mode

    Args:
        df: pandas.DataFrame, the dataframe containing the set infromation (filenames, labels),
            it should contain these columns :
            "filename"
            "filename", "event_labels"
            "filename", "onset", "offset", "event_label"
        get_feature_file_func: function(), function which take a filename as input and return a feature file
        encode_function: function(), function which encode labels
        transform: function(), (Default value = None), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, (Default value = False) whether or not to return indexes when use __getitem__

    Attributes:
        df: pandas.DataFrame, the dataframe containing the set infromation (filenames, labels, ...)
        get_feature_file_func: function(), function which take a filename as input and return a feature file
        encode_function: function(), function which encode labels
        transform : function(), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, whether or not to return indexes when use __getitem__
    """
    def __init__(self, df, get_feature_file_func, encode_function, transform=None,
                 return_indexes=False):

        self.df = df
        self.get_feature_file_func = get_feature_file_func
        self.encode_function = encode_function
        self.transform = transform
        self.return_indexes = return_indexes

    def set_return_indexes(self, val):
        """ Set the value of self.return_indexes

        Args:
            val : bool, whether or not to return indexes when use __getitem__
        """
        self.return_indexes = val

    def __len__(self):
        """
        Returns:
            int
                Length of the object
        """
        length = len(self.df)
        return length

    def get_sample(self, index):
        """From an index, get the features and the labels to create a sample

        Args:
            index: int, Index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array)

        """
        features = self.get_feature_file_func(self.df.iloc[index]["filename"])

        # event_labels means weak labels, event_label means strong labels
        if "event_labels" in self.df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
            if "event_labels" in self.df.columns:
                label = self.df.iloc[index]["event_labels"]
                if pd.isna(label):
                    label = []
                if type(label) is str:
                    if label == "":
                        label = []
                    else:
                        label = label.split(",")
            else:
                cols= ["onset", "offset", "event_label"]
                label = self.df.iloc[index][cols]
                if pd.isna(label["event_label"]):
                    label = []
        else:
            label = "empty"  # trick to have -1 for unlabeled data and concat them with labeled
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns))
        if index == 0:
            LOG.debug("label to encode: {}".format(label))
        if self.encode_function is not None:
            # labels are a list of string or list of list [[label, onset, offset]]
            y = self.encode_function(label)
            # if frames is not None, it does not work for strong labels,
            # because we will have all the labels and not only ones from the frame
            # Todo, change this function or put always weak labels when frames not None
        else:
            y = label
        sample = features, y
        return sample

    def __getitem__(self, index):
        """ Get a sample and transform it to be used in a model, use the transformations

        Args:
            index : int, index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array) or
            Tuple containing the features, the labels and the index (numpy.array, numpy.array, int)

        """
        sample = self.get_sample(index)

        if self.transform:
            sample = self.transform(sample)

        if self.return_indexes:
            sample = (sample, index)

        return sample

    def set_transform(self, transform):
        """Set the transformations used on a sample

        Args:
            transform: function(), the new transformations
        """
        self.transform = transform

    def add_transform(self, transform):
        if type(self.transform) is not Compose:
            raise TypeError("To add transform, the transform should already be a compose of transforms")
        transforms = self.transform.add_transform(transform)
        return DataLoadDf(self.df, self.get_feature_file_func, self.encode_function, transforms, self.return_indexes)


class GaussianNoise:
    """ Apply gaussian noise
        Args:
            nb_frames: int, the number of frames to match
        Attributes:
            nb_frames: int, the number of frames to match
        """

    def __init__(self, mean=0, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)
        # sample must be a tuple or a list, not apply on labels
        for k in range(len(sample) - 1):
            sample[k] = sample[k] + np.abs(np.random.normal(0, 0.5 ** 2, sample[k].shape))

        return sample


class ApplyLog(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """ Apply the transformation
        Args:

        sample: tuple, a sample defined by a DataLoad class

        Returns:
            tuple
            The transformed tuple
        """
        # sample must be a tuple or a list, first parts are input, then last element is label
        if type(sample) is tuple:
            sample = list(sample)
        for i in range(len(sample) - 1):
            sample[i] = librosa.amplitude_to_db(sample[i].T).T
        return sample


class PadOrTrunc:
    """ Pad or truncate a sequence given a number of frames
    Args:
        nb_frames: int, the number of frames to match
    Attributes:
        nb_frames: int, the number of frames to match
    """

    def __init__(self, nb_frames):
        self.nb_frames = nb_frames

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)
        # sample must be a tuple or a list
        for k in range(len(sample) - 1):
            sample[k] = pad_trunc_seq(sample[k], self.nb_frames)

        if len(sample[-1].shape) == 2:
            sample[-1] = pad_trunc_seq(sample[-1], self.nb_frames)

        return sample


class AugmentGaussianNoise:
    """ Pad or truncate a sequence given a number of frames
           Args:
               mean: float, mean of the Gaussian noise to add
           Attributes:
               std: float, std of the Gaussian noise to add
           """

    def __init__(self, mean=0, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        sample, label = sample

        noise = sample + np.abs(np.random.normal(0, 0.5 ** 2, sample.shape))

        return sample, noise, label


class AugmentShiftTruncFreq:
    """ Pad or truncate a sequence given a number of frames
           Args:
               mean: float, mean of the Gaussian noise to add
           Attributes:
               std: float, std of the Gaussian noise to add
           """

    def __init__(self, nshift=None):
        self.nshift = nshift

    def trunc(self, features):
        """ Apply a frequency trunc on the data

        Args:
            features: numpy.array, features to be modified
        Returns:
            numpy.ndarray
            Modified features
        """
        shift_value = np.random.randint(-self.nshift, self.nshift)
        if shift_value > 0:
            X_new = np.pad(features, ((0, 0), (shift_value, 0)), mode='constant')[:, :-shift_value]
        else:
            shift_value = -shift_value
            X_new = np.pad(features, ((0, 0), (0, shift_value)), mode='constant')[:, shift_value:]
        return X_new

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        sample, label = sample

        trans = self.trunc(sample)

        return sample, trans, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    Args:
        unsqueeze_axis: int, (Default value = None) add an dimension to the axis mentioned.
            Useful to add a channel axis to use CNN.
    Attributes:
        unsqueeze_axis: int, add an dimension to the axis mentioned.
            Useful to add a channel axis to use CNN.
    """
    def __init__(self, unsqueeze_axis=None):
        self.unsqueeze_axis = unsqueeze_axis

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample : tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)
        # sample must be a tuple or a list, first parts are input, then last element is label
        for i in range(len(sample)):
            sample[i] = torch.from_numpy(sample[i]).float()  # even labels (we don't loop until -1)

        for i in range(len(sample) - 1):
            if self.unsqueeze_axis is not None:
                sample[i] = sample[i].unsqueeze(self.unsqueeze_axis)

        return sample


class Normalize(object):
    """Normalize inputs
    Args:
        scaler: Scaler object, the scaler to be used to normalize the data
    Attributes:
        scaler : Scaler object, the scaler to be used to normalize the data
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)
        # sample must be a tuple or a list
        for k in range(len(sample) - 1):
            sample[k] = self.scaler.normalize(sample[k])

        return sample


# class GetEmbedding:
#     """Get an embedding from a CNN trained
#         Args:
#             cnn_model: derived nn.Module object, the model to get the embedding from.
#             nb_frames_to_reshape: int, reshape the input dividing the actual nb of frames by this number of frames and
#             apply the model on the created inputs and get back to the original size.
#         Attributes:
#             cnn_model: derived nn.Module object, the model to get the embedding from.
#             nb_frames_to_reshape: int, reshape the input dividing the actual nb of frames by this number of frames and
#             apply the model on the created inputs and get back to the original size.
#         """
#     def __init__(self, cnn_model, nb_frames_to_reshape=None):
#         self.triplet_model = cnn_model
#         self.nb_frames = nb_frames_to_reshape
#
#     def __call__(self, sample):
#         """ Apply the transformation
#         Args:
#             sample: tuple, a sample defined by a DataLoad class
#
#         Returns:
#             tuple
#             The transformed tuple
#         """
#         if type(sample) is tuple:
#             sample = list(sample)
#
#         for k in range(len(sample) - 1):
#             samp = sample[k]
#             if self.nb_frames:
#                 original_nb_frames = samp.shape[-2]
#                 samp = samp.unsqueeze(0)
#                 samp = change_view_frames(samp, self.nb_frames)
#
#             embed = self.triplet_model(samp)
#             if self.nb_frames is not None:
#                 embed = change_view_frames(embed, original_nb_frames)
#             embed = embed.squeeze(-1)
#             embed = embed.squeeze(0)
#             embed = embed.permute(1, 0)  # inverse frames and channel (frames, channel)
#
#             sample[k] = embed.detach()
#
#         return sample


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms: list of ``Transform`` objects, list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>> transforms.Scale(),
        >>> transforms.PadTrim(max_len=16000),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def add_transform(self, transform):
        t = self.transforms.copy()
        t.append(transform)
        return Compose(t)

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Args:
        datasets : sequence, list of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @property
    def cluster_indices(self):
        cluster_ind = []
        count = 0
        for size in self.cumulative_sizes:
            cluster_ind.append(range(count, size))
            count += size
        return cluster_ind

    def __init__(self, datasets, batch_sizes=None):
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        if batch_sizes is not None:
            assert len(batch_sizes) == len(datasets), "If batch_sizes given, should be equal to the number " \
                                                      "of datasets "
        self.batch_sizes = batch_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    @property
    def df(self):
        df = self.datasets[0].df
        for dataset in self.datasets[1:]:
            df = pd.concat([df, dataset.df], axis=0, ignore_index=True, sort=False)
        return df


class Subset(DataLoadDf):
    """
    Subset of a dataset to be used when separating in multiple subsets

    Args:
        dataload_df: DataLoadDf or similar, dataset to be split
    indices: sequence, list of indices to keep in this subset
    """
    def __init__(self, dataload_df, indices):
        self.indices = indices
        self.df = dataload_df.df.loc[indices].reset_index(inplace=False, drop=True)

        super(Subset, self).__init__(self.df, dataload_df.get_feature_file_func, dataload_df.encode_function,
                                     dataload_df.transform, dataload_df.return_indexes)

    def __getitem__(self, idx):
        return super(Subset, self).__getitem__(idx)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Args:
        dataset: Dataset, dataset to be split
        lengths: sequence, lengths of splits to be produced
    """
    # if ratio > 1:
    # 	raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = np.random.permutation(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(itertools.accumulate(lengths), lengths)]


def train_valid_split(dataset, validation_amount):
    valid_length = int(validation_amount * len(dataset))
    train_length = len(dataset) - valid_length

    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])
    return train_dataset, valid_dataset


class ClusterRandomSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size=None, shuffle=True):
        self.data_source = data_source
        if batch_size is not None:
            assert self.data_source.batch_sizes is None, "do not declare batch size in sampler " \
                                                         "if data source already got one"
            self.batch_sizes = [batch_size for _ in self.data_source.cluster_indices]
        else:
            self.batch_sizes = self.data_source.batch_sizes
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for j, cluster_indices in enumerate(self.data_source.cluster_indices):
            batches = [
                cluster_indices[i:i + self.batch_sizes[j]] for i in range(0, len(cluster_indices), self.batch_sizes[j])
            ]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_sizes[j]]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)

            # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class MultiStreamBatchSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_sizes, shuffle=True):
        self.data_source = data_source
        self.batch_sizes = batch_sizes
        l_bs = len(batch_sizes)
        nb_dataset = len(self.data_source.cluster_indices)
        assert l_bs == nb_dataset, "batch_sizes must be the same length as the number of datasets in " \
                                   "the source {} != {}".format(l_bs, nb_dataset)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            for i in range(len(self.batch_sizes)):
                self.data_source.cluster_indices[i] = np.random.permutation(self.data_source.cluster_indices[i])
        iterators = []
        for i in range(len(self.batch_sizes)):
            iterators.append(grouper(self.data_source.cluster_indices[i], self.batch_sizes[i]))

        return (sum(subbatch_ind, ()) for subbatch_ind in zip(*iterators))

    def __len__(self):
        val = np.inf
        for i in range(len(self.batch_sizes)):
            val = min(val, len(self.data_source.cluster_indices[i]) // self.batch_sizes[i])
        return val


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n

    return zip(*args)
