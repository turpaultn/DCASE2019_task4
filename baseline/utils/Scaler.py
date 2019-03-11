import time
import logging
import numpy as np
import torch
import json
from utils.Logger import LOG


class Scaler(object):
    """
    operates on one or multiple existing datasets and applies operations
    """

    def __init__(self):
        self.mean_ = None
        self.mean_of_square_ = None
        self.std_ = None

    # compute the mean incrementaly
    def mean(self, data, axis=-1):
        # -1 means have at the end a mean vector of the last dimension
        if axis == -1:
            mean = data
            while len(mean.shape) != 1:
                mean = np.mean(mean, axis=0, dtype=np.float64)
        else:
            mean = np.mean(data, axis=axis, dtype=np.float64)
        return mean

    # compute variance thanks to mean and mean of square
    def variance(self, mean, mean_of_square):
        return mean_of_square - mean**2

    def means(self, dataset):
        """
       Splits a dataset in to train test validation.
       :param dataset: dataset, from DataLoad class, each sample is an (X, y) tuple.
       """
        LOG.info('computing mean')
        start = time.time()

        shape = None

        counter = 0
        for sample in dataset:
            if type(sample) in [tuple, list] and len(sample)==2:
                batch_X, _ = sample
            else:
                batch_X = sample
            if type(batch_X) is torch.Tensor:
                batch_X_arr = batch_X.numpy()
            else:
                batch_X_arr = batch_X
            data_square = batch_X_arr ** 2
            counter += 1

            if shape is None:
                shape = batch_X_arr.shape
            else:
                if not batch_X_arr.shape == shape:
                    raise NotImplementedError("Not possible to add data with different shape in mean calculation yet")

            # assume first item will have shape info
            if self.mean_ is None:
                self.mean_ = self.mean(batch_X_arr, axis=-1)
            else:
                self.mean_ += self.mean(batch_X_arr, axis=-1)

            if self.mean_of_square_ is None:
                self.mean_of_square_ = self.mean(data_square, axis=-1)
            else:
                self.mean_of_square_ += self.mean(data_square, axis=-1)

        self.mean_ /= counter
        self.mean_of_square_ /= counter

        ## To be used if data different shape, but need to stop the iteration before.
        # rest = len(dataset) - i
        # if rest != 0:
        #     weight = rest / float(i + rest)
        #     X, y = dataset[-1]
        #     data_square = X ** 2
        #     mean = mean * (1 - weight) + self.mean(X, axis=-1) * weight
        #     mean_of_square = mean_of_square * (1 - weight) + self.mean(data_square, axis=-1) * weight

        LOG.debug('time to compute means: ' + str(time.time() - start))
        return self

    def std(self, variance):
        return np.sqrt(variance)

    def calculate_scaler(self, dataset):
        self.means(dataset)
        variance = self.variance(self.mean_, self.mean_of_square_)
        self.std_ = self.std(variance)

        return self.mean_, self.std_

    def normalize(self, batch):
        if type(batch) is torch.Tensor:
            batch_ = batch.numpy()
            batch_ = (batch_ - self.mean_) / self.std_
            return torch.Tensor(batch_)
        else:
            return (batch - self.mean_) / self.std_

    def state_dict(self):
        if type(self.mean_) is not np.ndarray:
            raise NotImplementedError("Save scaler only implemented for numpy array means_")

        dict_save = {"mean_": self.mean_.tolist(),
                     "mean_of_square_": self.mean_of_square_.tolist()}
        return dict_save

    def save(self, path):
        dict_save = self.state_dict()
        with open(path, "w") as f:
            json.dump(dict_save, f)

    def load(self, path):
        with open(path, "r") as f:
            dict_save = json.load(f)

        self.load_state_dict(dict_save)

    def load_state_dict(self, state_dict):
        self.mean_ = np.array(state_dict["mean_"])
        self.mean_of_square_ = np.array(state_dict["mean_of_square_"])
        variance = self.variance(self.mean_, self.mean_of_square_)
        self.std_ = self.std(variance)
