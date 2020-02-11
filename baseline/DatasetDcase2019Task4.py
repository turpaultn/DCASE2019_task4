# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

from __future__ import print_function
import numpy as np
import os
import librosa
import time
import pandas as pd

import config as cfg
from utils.Logger import LOG
from download_data import download
from utils.utils import create_folder, read_audio


class DatasetDcase2019Task4:
    """DCASE 2018 task 4 dataset
    This dataset contains multiple subsets:
    A train set divided in three subsets:
        - A weakly labeled set
        - An unlabeled-in-domain set
        - An unlabeled-out-of-domain set
    A test set
    An evaluation set
    The files should be ordered into the 'local_path' as described here:
    dataset root
        - readme.md
        - download_data.py
        - metadata
            - train
                - weak.tsv
                - unlabel_in_domain.tsv
                - synthetic_data.tsv
            - validation
                - validation.tsv
                - test_dcase2018.tsv
                - eval_dcase2018.tsv
            -eval
                - public.tsv
        - audio
            - train
                - weak
                - unlabel_in_domain
                - synthetic_data
            - validation
            - eval
                - public

    Args:
        local_path: str, (Default value = "") base directory where the dataset is, to be changed if
            dataset moved
        base_feature_dir: str, (Default value = "features) base directory to store the features
        recompute_features: bool, (Default value = False) wether or not to recompute features
        subpart_data: int, (Default value = None) allow to take only a small part of the dataset.
            This number represents the number of data to download and use from each set
        save_log_feature: bool, (Default value = True) whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)

    Attributes:
        local_path: str, base directory where the dataset is, to be changed if
            dataset moved
        base_feature_dir: str, base directory to store the features
        recompute_features: bool, wether or not to recompute features
        subpart_data: int, allow to take only a small part of the dataset.
            This number represents the number of data to download and use from each set
        save_log_feature: bool, whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)
        feature_dir : str, directory to store the features

    """
    def __init__(self, local_path="", base_feature_dir="features", recompute_features=False,
                 save_log_feature=True):

        self.local_path = local_path
        self.recompute_features = recompute_features
        self.save_log_feature = save_log_feature

        feature_dir = os.path.join(base_feature_dir, "sr" + str(cfg.sample_rate) + "_win" + str(cfg.n_window)
                                   + "_hop" + str(cfg.hop_length) + "_mels" + str(cfg.n_mels))
        if not self.save_log_feature:
            feature_dir += "_nolog"

        self.feature_dir = os.path.join(feature_dir, "features")
        # create folder if not exist
        create_folder(self.feature_dir)

    def initialize_and_get_df(self, tsv_path, subpart_data=None, download=True):
        """ Initialize the dataset, extract the features dataframes
        Args:
            tsv_path: str, tsv path in the initial dataset
            subpart_data: int, the number of file to take in the dataframe if taking a small part of the dataset.
            download: bool, whether or not to download the data from the internet (youtube).

        Returns:
            pd.DataFrame
            The dataframe containing the right features and labels
        """
        meta_name = os.path.join(self.local_path, tsv_path)
        if download:
            self.download_from_meta(meta_name, subpart_data)
        return self.extract_features_from_meta(meta_name, subpart_data)

    @staticmethod
    def get_classes(list_dfs):
        """ Get the different classes of the dataset
        Returns:
            A list containing the classes
        """
        classes = []
        for df in list_dfs:
            if "event_label" in df.columns:
                classes.extend(df["event_label"].dropna().unique())  # dropna avoid the issue between string and float
            elif "event_labels" in df.columns:
                classes.extend(df.event_labels.str.split(',', expand=True).unstack().dropna().unique())
        return list(set(classes))

    @staticmethod
    def get_subpart_data(df, subpart_data):
        column = "filename"
        if not subpart_data > len(df[column].unique()):
            filenames = df[column].drop_duplicates().sample(subpart_data, random_state=10)
            df = df[df[column].isin(filenames)].reset_index(drop=True)
            LOG.debug("Taking subpart of the data, len : {}, df_len: {}".format(subpart_data, len(df)))
        return df

    @staticmethod
    def get_df_from_meta(meta_name, subpart_data=None):
        """
        Extract a pandas dataframe from a tsv file

        Args:
            meta_name : str, path of the tsv file to extract the df
            subpart_data: int, the number of file to take in the dataframe if taking a small part of the dataset.

        Returns:
            dataframe
        """
        df = pd.read_csv(meta_name, header=0, sep="\t")
        if subpart_data is not None:
            df = DatasetDcase2019Task4.get_subpart_data(df, subpart_data)
        return df

    @staticmethod
    def get_audio_dir_path_from_meta(filepath):
        """ Get the corresponding audio dir from a meta filepath

        Args:
            filepath : str, path of the meta filename (tsv)

        Returns:
            str
            path of the audio directory.
        """
        base_filepath = os.path.splitext(filepath)[0]
        audio_dir = base_filepath.replace("metadata", "audio")
        if audio_dir.split('/')[-2] in ['validation']:
            audio_dir = '/'.join(audio_dir.split('/')[:-1])
        audio_dir = os.path.abspath(audio_dir)
        return audio_dir

    def download_from_meta(self, filename, subpart_data=None, n_jobs=3, chunk_size=10):
        """
        Download files contained in a meta file (tsv)

        Args:
            filename: str, path of the meta file containing the name of audio files to donwnload
                (tsv with column "filename")
            subpart_data: int, the number of files to use, if a subpart of the dataframe wanted.
            chunk_size: int, (Default value = 10) number of files to download in a chunk
            n_jobs : int, (Default value = 3) number of parallel jobs
        """
        result_audio_directory = self.get_audio_dir_path_from_meta(filename)
        # read metadata file and get only one filename once
        df = DatasetDcase2019Task4.get_df_from_meta(filename, subpart_data)
        filenames = df.filename.drop_duplicates()
        download(filenames, result_audio_directory, n_jobs=n_jobs, chunk_size=chunk_size)

    def get_feature_file(self, filename):
        """
        Get a feature file from a filename
        Args:
            filename:  str, name of the file to get the feature

        Returns:
            numpy.array
            containing the features computed previously
        """
        fname = os.path.join(self.feature_dir, os.path.splitext(filename)[0] + ".npy")
        data = np.load(fname)
        return data

    def calculate_mel_spec(self, audio):
        """
        Calculate a mal spectrogram from raw audio waveform
        Note: The parameters of the spectrograms are in the config.py file.
        Args:
            audio : numpy.array, raw waveform to compute the spectrogram

        Returns:
            numpy.array
            containing the mel spectrogram
        """
        # Compute spectrogram
        ham_win = np.hamming(cfg.n_window)

        spec = librosa.stft(
            audio,
            n_fft=cfg.n_window,
            hop_length=cfg.hop_length,
            window=ham_win,
            center=True,
            pad_mode='reflect'
        )

        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
            sr=cfg.sample_rate,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min, fmax=cfg.f_max,
            htk=False, norm=None)

        if self.save_log_feature:
            mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
        mel_spec = mel_spec.T
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def extract_features_from_meta(self, tsv_audio, subpart_data=None):
        """Extract log mel spectrogram features.

        Args:
            tsv_audio : str, file containing names, durations and labels : (name, start, end, label, label_index)
                the associated wav_filename is Yname_start_end.wav
            subpart_data: int, number of files to extract features from the tsv.
        """
        t1 = time.time()
        df_meta = self.get_df_from_meta(tsv_audio, subpart_data)
        LOG.info("{} Total file number: {}".format(tsv_audio, len(df_meta.filename.unique())))

        for ind, wav_name in enumerate(df_meta.filename.unique()):
            if ind % 500 == 0:
                LOG.debug(ind)
            wav_dir = self.get_audio_dir_path_from_meta(tsv_audio)
            wav_path = os.path.join(wav_dir, wav_name)

            out_filename = os.path.splitext(wav_name)[0] + ".npy"
            out_path = os.path.join(self.feature_dir, out_filename)

            if not os.path.exists(out_path):
                if not os.path.isfile(wav_path):
                    LOG.error("File %s is in the tsv file but the feature is not extracted!" % wav_path)
                    df_meta = df_meta.drop(df_meta[df_meta.filename == wav_name].index)
                else:
                    (audio, _) = read_audio(wav_path, cfg.sample_rate)
                    if audio.shape[0] == 0:
                        print("File %s is corrupted!" % wav_path)
                    else:
                        mel_spec = self.calculate_mel_spec(audio)

                        np.save(out_path, mel_spec)

                    LOG.debug("compute features time: %s" % (time.time() - t1))

        return df_meta.reset_index(drop=True)
