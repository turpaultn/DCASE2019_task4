# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################


import argparse
import os
import pandas as pd
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from DatasetDcase2019Task4 import DatasetDcase2019Task4
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from utils.Scaler import Scaler
from TestModel import test_model
from evaluation_measures import get_f_measure_by_class, get_predictions, audio_tagging_results, compute_strong_metrics
from models.CRNN import CRNN
import config as cfg
from utils.utils import ManyHotEncoder, AverageMeterSet, create_folder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms
from utils.Logger import LOG


def train(train_loader, model, optimizer, epoch, weak_mask=None, strong_mask=None):
    class_criterion = nn.BCELoss()
    [class_criterion] = to_cuda_if_available([class_criterion])

    meters = AverageMeterSet()
    meters.update('lr', optimizer.param_groups[0]['lr'])

    LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    for i, (batch_input, target) in enumerate(train_loader):
        [batch_input, target] = to_cuda_if_available([batch_input, target])
        LOG.debug(batch_input.mean())

        strong_pred, weak_pred = model(batch_input)
        loss = 0
        if weak_mask is not None:
            # Weak BCE Loss
            # Trick to not take unlabeled data
            # Todo figure out another way
            target_weak = target.max(-2)[0]
            weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
            if i == 1:
                LOG.debug("target: {}".format(target.mean(-2)))
                LOG.debug("Target_weak: {}".format(target_weak))
                LOG.debug(weak_class_loss)
            meters.update('Weak loss', weak_class_loss.item())

            loss += weak_class_loss

        if strong_mask is not None:
            # Strong BCE loss
            strong_class_loss = class_criterion(strong_pred[strong_mask], target[strong_mask])
            meters.update('Strong loss', strong_class_loss.item())

            loss += strong_class_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_time = time.time() - start

    LOG.info(
        'Epoch: {}\t'
        'Time {:.2f}\t'
        '{meters}'.format(
            epoch, epoch_time, meters=meters))


if __name__ == '__main__':
    LOG.info("Simple CRNNs")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")
    parser.add_argument("-n", '--no_weak', dest='no_weak', action='store_true', default=False,
                        help="Not using weak labels during training")
    f_args = parser.parse_args()

    reduced_number_of_data = f_args.subpart_data
    no_weak = f_args.no_weak
    LOG.info("subpart_data = {}".format(reduced_number_of_data))
    LOG.info("Using_weak labels : {}".format(not no_weak))

    if not no_weak:
        add_dir_path = "_with_weak"
    else:
        add_dir_path = "_synthetic_only"

    store_dir = os.path.join("stored_data", "simple_CRNN" + add_dir_path)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    create_folder(store_dir)
    create_folder(saved_model_dir)
    create_folder(saved_pred_dir)

    # ##############
    # Model
    # ##############

    crnn_kwargs = cfg.crnn_kwargs
    crnn = CRNN(**crnn_kwargs)
    crnn.apply(weights_init)
    pooling_time_ratio = cfg.pooling_time_ratio

    LOG.info(crnn)

    # ##############
    # DATA
    # ##############
    dataset = DatasetDcase2019Task4(os.path.join(cfg.workspace),
                                    base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                                    save_log_feature=False)

    weak_df = dataset.initialize_and_get_df(cfg.weak, reduced_number_of_data)
    synthetic_df = dataset.initialize_and_get_df(cfg.synthetic, reduced_number_of_data, download=False)
    validation_df = dataset.initialize_and_get_df(cfg.validation, reduced_number_of_data)

    classes = DatasetDcase2019Task4.get_classes([weak_df, validation_df, synthetic_df])

    # Be careful, frames is max_frames // pooling_time_ratio because max_pooling is applied on time axis in the model
    many_hot_encoder = ManyHotEncoder(classes, n_frames=cfg.max_frames // pooling_time_ratio)

    transforms = get_transforms(cfg.max_frames)

    # Divide weak in train and valid
    train_weak_df = weak_df.sample(frac=0.8, random_state=26)
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    LOG.debug(valid_weak_df.event_labels.value_counts())
    train_weak_data = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                 transform=transforms)

    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)

    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df_frames = train_synth_df.copy()
    train_synth_df_frames.onset = train_synth_df_frames.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    train_synth_df_frames.offset = train_synth_df_frames.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    LOG.debug(valid_synth_df.event_label.value_counts())
    LOG.debug(valid_synth_df)
    train_synth_data = DataLoadDf(train_synth_df_frames, dataset.get_feature_file,
                                  many_hot_encoder.encode_strong_df,
                                  transform=transforms)

    if not no_weak:
        list_datasets = [train_weak_data, train_synth_data]
        training_data = ConcatDataset(list_datasets)
    else:
        list_datasets = [train_synth_data]
        training_data = train_synth_data

    scaler = Scaler()
    scaler.calculate_scaler(training_data)
    LOG.debug(scaler.mean_)

    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)
    # Validation dataset is only used to get an idea of wha could be results on evaluation dataset
    validation_dataset = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                    transform=transforms_valid)

    transforms = get_transforms(cfg.max_frames, scaler)
    train_synth_data.set_transform(transforms)
    if not no_weak:
        train_weak_data.set_transform(transforms)
        concat_dataset = ConcatDataset([train_weak_data, train_synth_data])
        # Taking as much data from synthetic than strong.
        sampler = MultiStreamBatchSampler(concat_dataset,
                                          batch_sizes=[cfg.batch_size // 2, cfg.batch_size // 2])
        training_data = DataLoader(concat_dataset, batch_sampler=sampler)
        valid_weak_data = DataLoadDf(valid_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                     transform=transforms_valid)
        weak_mask = slice(cfg.batch_size // 2)
        strong_mask = slice(cfg.batch_size // 2, cfg.batch_size)
    else:
        training_data = DataLoader(train_synth_data, batch_size=cfg.batch_size)
        strong_mask = slice(cfg.batch_size)  # Not masking
        weak_mask = None

    valid_synth_data = DataLoadDf(valid_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms_valid)

    # ##############
    # Train
    # ##############
    optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    LOG.info(optimizer)
    bce_loss = nn.BCELoss()

    state = {
        'model': {"name": crnn.__class__.__name__,
                  'args': '',
                  "kwargs": crnn_kwargs,
                  'state_dict': crnn.state_dict()},
        'optimizer': {"name": optimizer.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': optimizer.state_dict()},
        "pooling_time_ratio": pooling_time_ratio,
        'scaler': scaler.state_dict(),
        "many_hot_encoder": many_hot_encoder.state_dict()
    }

    save_best_cb = SaveBest("sup")

    # Eval 2018
    eval_2018_df = dataset.initialize_and_get_df(cfg.eval2018, reduced_number_of_data)
    eval_2018 = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                           transform=transforms_valid)

    [crnn] = to_cuda_if_available([crnn])
    for epoch in range(cfg.n_epoch):
        crnn = crnn.train()

        train(training_data, crnn, optimizer, epoch, weak_mask, strong_mask)

        crnn = crnn.eval()
        LOG.info("Training synthetic metric:")
        train_predictions = get_predictions(crnn, train_synth_data, many_hot_encoder.decode_strong, pooling_time_ratio,
                                            save_predictions=None)
        train_metric = compute_strong_metrics(train_predictions, train_synth_df)

        if not no_weak:
            LOG.info("Training weak metric:")
            weak_metric = get_f_measure_by_class(crnn, len(classes),
                                                 DataLoader(train_weak_data, batch_size=cfg.batch_size))
            LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
            LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

            LOG.info("Valid weak metric:")
            weak_metric = get_f_measure_by_class(crnn, len(classes),
                                                 DataLoader(valid_weak_data, batch_size=cfg.batch_size))

            LOG.info(
                "Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
            LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

        LOG.info("Valid synthetic metric:")
        predictions = get_predictions(crnn, valid_synth_data, many_hot_encoder.decode_strong, pooling_time_ratio)
        valid_metric = compute_strong_metrics(predictions, valid_synth_df)

        state['model']['state_dict'] = crnn.state_dict()
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_metric.results()
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)
            print("Saving this model: {}".format(model_fname))

        if cfg.save_best:
            global_valid = valid_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
            if not no_weak:
                global_valid += np.mean(weak_metric)
            if save_best_cb.apply(global_valid):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)

    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        LOG.info("testing model: {}".format(model_fname))
    else:
        LOG.info("testing model of last epoch: {}".format(cfg.n_epoch))

    # ##############
    # Validation
    # ##############
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")
    test_model(state, cfg.validation, reduced_number_of_data, predicitons_fname)

    # ##############
    # Evaluation
    # ##############
    predicitons_eval2019_fname = os.path.join(saved_pred_dir, "baseline_eval2019.tsv")
    test_model(state, cfg.eval_desed, reduced_number_of_data, predicitons_eval2019_fname)
