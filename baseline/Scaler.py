import time
import logging
import numpy as np
import torch
import config as cfg

logger = logging.getLogger("sed")


class Scaler(object):
    """
    operates on one or multiple existing datasets and applies operations
    """

    def __init__(self):
        self.mean_ = None
        self.mean_of_square_ = None

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
        logger.info('computing mean')
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

        logger.debug('time to compute means: ' + str(time.time() - start))
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


if __name__ == '__main__':
    from DatasetDcase2019Task4 import DatasetDcase2018Task4
    from utils import ManyHotEncoder
    from DataLoad import ToTensor, DataLoadDf
    import math
    from torch.utils.data import DataLoader

    dataset = DatasetDcase2018Task4("dcase2018", base_feature_dir="dcase2018/features")

    batch_size = 4
    max_frames = math.ceil(10 * cfg.sample_rate / cfg.hop_length)  # assuming max 10 sec
    n_epoch = 10
    classes = np.unique(dataset.get_df_from_meta(dataset.test)["event_label"])
    n_class = len(classes)

    test_df = dataset.initialize_and_get_df(dataset.test, max_frames)
    many_hot_encoder = ManyHotEncoder(classes, n_frames=max_frames)
    try_dataset = DataLoadDf(test_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                 transform=ToTensor())
    dataloader = DataLoader(try_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    scaler = Scaler()
    scaler.calculate_scaler(dataloader)
    print(scaler.mean_)
    print(scaler.std_)
    try_dataset.transform = ToTensor(unsqueeze_axis=0)
    # train_loader = DataLoader(weak_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(scaler.normalize(next(iter(dataloader))[0]))