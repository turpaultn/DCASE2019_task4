import warnings

import torch.nn as nn
import torch

from models.RNN import BidirectionalGRU
from models.CNN import CNN


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class CRNN(nn.Module):

    def __init__(self, n_in_channel, nclass, activation="Relu", mode="strong", dropout=0,
                 train_cnn=True, max_frames=None, rnn_type='BGRU', n_layers_RNN=1, dropout_recurrent=0, **kwargs):
        super(CRNN, self).__init__()

        self.mode = mode
        self.max_frames = max_frames

        self.cnn = CNN(n_in_channel, activation, dropout, **kwargs)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn
        if rnn_type == 'BGRU':
            self.rnn = BidirectionalGRU(64, 32, dropout=dropout_recurrent, num_layers=n_layers_RNN)
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(64, nclass)
        self.sigmoid = nn.Sigmoid()

    def load_cnn(self, parameters):
        self.cnn.load(parameters)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load(self, filename=None, parameters=None):
        if filename is not None:
            parameters = torch.load(filename)
        if parameters is None:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        self.cnn.load(parameters=parameters["cnn"])
        self.rnn.load_state_dict(parameters["rnn"])
        self.dense.load_state_dict(parameters["dense"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn": self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(), 'rnn': self.rnn.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if freq != 1:
            warnings.warn("Output shape is: {}".format((bs, frames, chan * freq)))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        if self.max_frames is not None:
            x = x.contiguous().view(-1, self.max_frames, chan*freq)

        # rnn features
        x = self.rnn(x)
        x = self.dropout(x)
        x = self.dense(x)  # [bs, frames, nclass]
        x = self.sigmoid(x)

        return x


if __name__ == '__main__':
    CRNN(64, 10, kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
         pooling=[(1, 4), (1, 4), (1, 4)])
