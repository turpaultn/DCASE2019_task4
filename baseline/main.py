# -*- coding: utf-8 -*-
#########################################################################
# This file is derived from Curious AI/mean-teacher, under the Creative Commons Attribution-NonCommercial
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

import argparse
import os
import time

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

import ramps
from DatasetDcase2019Task4 import DatasetDcase2019Task4
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from Scaler import Scaler
from evaluation_measures import get_f_measure_by_class, get_predictions, audio_tagging_results, compute_strong_metrics
from models.CRNN import CRNN
import config as cfg
from utils import ManyHotEncoder, create_folder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms, AverageMeterSet
from torch import nn
from Logger import LOG


def adjust_learning_rate(optimizer, rampup_value, rampdown_value):
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, beta2)
        param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, optimizer, epoch, ema_model=None, weak_mask=None, strong_mask=None):
    """ One epoch of a Mean Teacher model
    :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
    Should return 3 values: teacher input, student input, labels
    :param model: torch.Module, model to be trained, should return a weak and strong prediction
    :param optimizer: torch.Module, optimizer used to train the model
    :param epoch: int, the current epoch of training
    :param ema_model: torch.Module, student model, should return a weak and strong prediction
    :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
    :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
    """
    class_criterion = nn.BCELoss()
    consistency_criterion_strong = nn.MSELoss()
    [class_criterion, consistency_criterion_strong] = to_cuda_if_available(
        [class_criterion, consistency_criterion_strong])

    meters = AverageMeterSet()

    LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    rampup_length = len(train_loader) * cfg.n_epoch // 2
    for i, (batch_input, ema_batch_input, target) in enumerate(train_loader):
        global_step = epoch + i
        if global_step < rampup_length:
            rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        else:
            rampup_value = 1.0

        # Todo check if this improves the performance
        # adjust_learning_rate(optimizer, rampup_value, rampdown_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        [batch_input, ema_batch_input, target] = to_cuda_if_available([batch_input, ema_batch_input, target])
        LOG.debug(batch_input.mean())
        # Outputs
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()

        strong_pred, weak_pred = model(batch_input)
        loss = None
        if weak_mask is not None:
            # Weak BCE Loss
            # Take the max in the time axis
            target_weak = target.max(-2)[0]
            weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
            if i == 0:
                LOG.debug("target: {}".format(target.mean(-2)))
                LOG.debug("Target_weak: {}".format(target_weak))
                LOG.debug("Target_weak mask: {}".format(target_weak[strong_mask]))
                LOG.debug(weak_class_loss)
                LOG.debug("rampup_value: {}".format(rampup_value))
            meters.update('weak_class_loss', weak_class_loss.item())

            ema_class_loss = class_criterion(weak_pred_ema[strong_mask], target_weak[strong_mask])
            meters.update('Weak EMA loss', ema_class_loss.item())

            loss = weak_class_loss

        # Strong BCE loss
        if strong_mask:
            strong_class_loss = class_criterion(strong_pred[strong_mask], target[strong_mask])
            meters.update('Strong loss', strong_class_loss.item())

            strong_ema_class_loss = class_criterion(strong_pred_ema[strong_mask], target[strong_mask])
            meters.update('Strong EMA loss', strong_ema_class_loss.item())
            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:

            consistency_cost = cfg.max_consistency_cost * rampup_value
            if cfg.max_consistency_cost is not None:
                meters.update('Consistency weight', consistency_cost)
                # Take only the consistence with weak and unlabel
                consistency_loss_strong = consistency_cost * consistency_criterion_strong(strong_pred[strong_mask],
                                                                                          strong_pred_ema[strong_mask])
                meters.update('Consistency loss', consistency_loss_strong.item())
                if loss is not None:
                    loss += consistency_loss_strong
                else:
                    loss = consistency_loss_strong

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start

    LOG.info(
        'Epoch: {}\t'
        'Time {:.2f}\t'
        '{meters}'.format(
            epoch, epoch_time, meters=meters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                        help="Not using synthetic labels during training")
    f_args = parser.parse_args()

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic
    LOG.info("subpart_data = {}".format(reduced_number_of_data))
    LOG.info("no_synthetic = {}".format(no_synthetic))

    if no_synthetic:
        add_dir_model_name = "_no_synthetic"
    else:
        add_dir_model_name = "_with_synthetic"

    store_dir = os.path.join("stored_data", "MeanTeacher" + add_dir_model_name)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    scaler_path = os.path.join(store_dir, "scaler")
    create_folder(store_dir)
    create_folder(saved_model_dir)
    create_folder(saved_pred_dir)

    pooling_time_ratio = cfg.pooling_time_ratio  # --> Be careful, it depends of the model time axis pooling
    # ##############
    # DATA
    # ##############
    dataset = DatasetDcase2019Task4(cfg.workspace,
                                    base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                                    save_log_feature=False)

    weak_df = dataset.intialize_and_get_df(cfg.weak, reduced_number_of_data)
    unlabel_df = dataset.intialize_and_get_df(cfg.unlabel, reduced_number_of_data)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = dataset.intialize_and_get_df(cfg.synthetic, reduced_number_of_data, download=False)
    validation_df = dataset.intialize_and_get_df(cfg.validation, reduced_number_of_data)

    classes = cfg.classes
    many_hot_encoder = ManyHotEncoder(classes, n_frames=cfg.max_frames // pooling_time_ratio)

    transforms = get_transforms(cfg.max_frames)

    # Divide weak in train and valid
    train_weak_df = weak_df.sample(frac=0.8, random_state=26)
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    LOG.debug(valid_weak_df.event_labels.value_counts())

    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)

    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    LOG.debug(valid_synth_df.event_label.value_counts())

    train_weak_data = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                 transform=transforms)
    unlabel_data = DataLoadDf(unlabel_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                              transform=transforms)
    train_synth_data = DataLoadDf(train_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms)

    if not no_synthetic:
        list_dataset = [train_weak_data, unlabel_data, train_synth_data]
        batch_sizes = [cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4]
        strong_mask = slice(cfg.batch_size//4 + cfg.batch_size//2, cfg.batch_size)
    else:
        list_dataset = [train_weak_data, unlabel_data]
        batch_sizes = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
        strong_mask = None
    # Assume weak data is always the first one
    weak_mask = slice(batch_sizes[0])

    scaler = Scaler()
    scaler.calculate_scaler(ConcatDataset(list_dataset))
    scaler.save(scaler_path)

    LOG.debug(scaler.mean_)

    transforms = get_transforms(cfg.max_frames, scaler, augment_type="noise")
    for i in range(len(list_dataset)):
        list_dataset[i].set_transform(transforms)

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset,
                                      batch_sizes=batch_sizes)

    training_data = DataLoader(concat_dataset, batch_sampler=sampler)

    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)
    valid_synth_data = DataLoadDf(valid_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms_valid)
    valid_weak_data = DataLoadDf(valid_weak_df, dataset.get_feature_file, many_hot_encoder.encode_weak,
                                  transform=transforms_valid)

    # Eval 2018
    eval_2018_df = dataset.intialize_and_get_df(cfg.eval2018, reduced_number_of_data)
    eval_2018 = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                           transform=transforms_valid)

    # ##############
    # Model
    # ##############
    crnn_kwargs = cfg.crnn_kwargs
    crnn = CRNN(**crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

    crnn.apply(weights_init)
    crnn_ema.apply(weights_init)
    LOG.info(crnn)

    for param in crnn_ema.parameters():
        param.detach_()

    optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    bce_loss = nn.BCELoss()

    state = {
        'model': {"name": crnn.__class__.__name__,
                  'args': '',
                  "kwargs": crnn_kwargs,
                  'state_dict': crnn.state_dict()},
        'model_ema': {"name": crnn_ema.__class__.__name__,
                      'args': '',
                      "kwargs": crnn_kwargs,
                      'state_dict': crnn_ema.state_dict()},
        'optimizer': {"name": optimizer.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': optimizer.state_dict()},
        "pooling_time_ratio": pooling_time_ratio
    }

    save_best_cb = SaveBest("sup")

    # ##############
    # Train
    # ##############
    for epoch in range(cfg.n_epoch):
        crnn = crnn.train()
        crnn_ema = crnn_ema.train()

        [crnn, crnn_ema] = to_cuda_if_available([crnn, crnn_ema])

        train(training_data, crnn, optimizer, epoch, ema_model=crnn_ema, weak_mask=weak_mask, strong_mask=strong_mask)

        crnn = crnn.eval()
        LOG.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn, valid_synth_data, many_hot_encoder.decode_strong,
                                      save_predictions=None)
        valid_events_metric = compute_strong_metrics(predictions, valid_synth_df, pooling_time_ratio)

        #Eval 2018
        LOG.info("\n ### Eval 2018 metric ### \n")
        predictions = get_predictions(crnn, eval_2018, many_hot_encoder.decode_strong,
                                      save_predictions=None)
        eval_2018_events_metric = compute_strong_metrics(predictions, eval_2018_df, pooling_time_ratio)

        LOG.info("\n ### Valid weak metric ### \n")
        weak_metric = get_f_measure_by_class(crnn, len(classes),
                                             DataLoader(valid_weak_data, batch_size=cfg.batch_size))

        LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
        LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

        state['model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_events_metric.results()
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            if not no_synthetic:
                global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
                global_valid = global_valid + np.mean(weak_metric)
            else:
                global_valid = np.mean(weak_metric)
            if save_best_cb.apply(global_valid):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)

    if cfg.save_best:
        state = torch.load(os.path.join(saved_model_dir, "baseline_best"))
        crnn.load(parameters=state["model"]["state_dict"])

    # ##############
    # Validation
    # ##############
    crnn = crnn.eval()
    scaler = Scaler()
    scaler.load(scaler_path)
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)

    # Eval 2018
    eval_2018_df = dataset.intialize_and_get_df(cfg.eval2018, reduced_number_of_data)
    # Strong
    eval_2018_strong = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                           transform=transforms_valid)
    predictions = get_predictions(crnn, eval_2018_strong, many_hot_encoder.decode_strong, save_predictions=None)
    compute_strong_metrics(predictions, eval_2018_df, pooling_time_ratio)
    # Weak
    eval_2018_weak = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_weak,
                                transform=transforms_valid)
    weak_metric = get_f_measure_by_class(crnn, len(classes), DataLoader(eval_2018_weak, batch_size=cfg.batch_size))
    LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    # Validation 2019
    validation_strong = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                   transform=transforms_valid)
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.csv")
    predictions = get_predictions(crnn, validation_strong, many_hot_encoder.decode_strong,
                                  save_predictions=predicitons_fname)
    compute_strong_metrics(predictions, validation_df, pooling_time_ratio)

    validation_weak = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_weak,
                                 transform=transforms_valid)
    weak_metric = get_f_measure_by_class(crnn, len(classes), DataLoader(validation_weak, batch_size=cfg.batch_size))
    LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    print(audio_tagging_results(validation_df, predictions))
