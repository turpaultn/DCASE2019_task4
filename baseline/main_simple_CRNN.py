import argparse
import os
import pandas as pd
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from DatasetDcase2019Task4 import DatasetDcase2019Task4
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from Scaler import Scaler
from evaluation_measures import event_based_evaluation_df, get_f_measure_by_class, segment_based_evaluation_df, \
    get_predictions
from models.CRNN import CRNN
import config as cfg
from utils import ManyHotEncoder, AverageMeterSet, create_folder, SaveBest, to_cuda_if_available, weights_init
from torch import nn
from Logger import LOG


def train(train_loader, model, optimizer, epoch):
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

        # Weak BCE Loss
        # Trick to not take unlabeled data
        # Todo figure out another way
        target_weak = target.max(-2)[0]
        weak_size = train_loader.batch_sampler.batch_sizes[0]
        weak_class_loss = class_criterion(weak_pred[:weak_size], target_weak[:weak_size])
        if i == 1:
            LOG.debug("target: {}".format(target.mean(-2)))
            LOG.debug("Target_weak: {}".format(target_weak))
            LOG.debug(weak_class_loss)
        meters.update('weak_class_loss', weak_class_loss.item())

        loss = weak_class_loss

        # Strong BCE loss
        strong_size = train_loader.batch_sampler.batch_sizes[-1]
        strong_class_loss = class_criterion(strong_pred[-strong_size:], target[-strong_size:])
        meters.update('strong_class_loss', strong_class_loss.item())

        loss += strong_class_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    meters.update('epoch_time', time.time() - start)

    LOG.info(
        'Epoch: {}\t'
        'Time {meters[epoch_time]:.2f}\t'
        'LR {meters[lr]:.2E}\t'
        'Loss {meters[loss]:.4f}\t'
        'Weak_loss {meters[weak_class_loss]:.4f}\t'
        'Strong_loss {meters[strong_class_loss]:.4f}\t'
        ''.format(
            epoch, meters=meters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")
    parser.add_argument("-m", '--model_path', type=str, default=None, dest="model_path",
                        help="Path of the model to be resume or to get validation results from.")
    parser.add_argument("-r", '--resume', type=bool, default=False, dest="resume",
                        help="Whether or not resuming the model given in model_path.")
    f_args = parser.parse_args()

    reduced_number_of_data = f_args.subpart_data
    model_path = f_args.model_path
    resume = f_args.resume
    LOG.info("subpart_data = {}".format(reduced_number_of_data))
    LOG.info("Loading model from = {}".format(model_path))
    LOG.info("Continue training : {}".format(resume))
    max_frames = cfg.max_frames

    store_dir = "stored_data"
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    scaler_path = os.path.join(store_dir, "scaler")
    create_folder(store_dir)
    create_folder(saved_model_dir)
    create_folder(saved_pred_dir)

    # ##############
    # DATA
    # ##############
    dataset = DatasetDcase2019Task4(os.path.join(".."),
                                    base_feature_dir=os.path.join("..", "dataset", "features"),
                                    subpart_data=reduced_number_of_data,
                                    save_log_feature=False)

    weak_df = dataset.intialize_and_get_df(cfg.weak, reduced_number_of_data)
    synthetic_df = dataset.intialize_and_get_df(cfg.synthetic, reduced_number_of_data, download=False)
    validation_df = dataset.intialize_and_get_df(cfg.validation, reduced_number_of_data)

    classes = DatasetDcase2019Task4.get_classes([weak_df, validation_df, synthetic_df])

    pooling_time_ratio = 8
    # Be careful, frames is max_frames // 8 because max_pooling is applied on time axis in the model
    many_hot_encoder = ManyHotEncoder(classes, n_frames=max_frames // pooling_time_ratio)

    transforms = get_transforms(max_frames)

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
    train_synth_df_frames = train_synth_df.copy()
    train_synth_df_frames.onset = train_synth_df_frames.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    train_synth_df_frames.offset = train_synth_df_frames.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    LOG.debug(valid_synth_df.event_label.value_counts())

    LOG.debug(valid_synth_df)

    train_weak_data = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                 transform=transforms)
    train_synth_data = DataLoadDf(train_synth_df_frames, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms)
    scaler = Scaler()
    scaler.calculate_scaler(ConcatDataset([train_weak_data, train_synth_data]))
    scaler.save(scaler_path)
    LOG.debug(scaler.mean_)

    transforms_valid = get_transforms(max_frames, scaler=scaler)
    # Validation dataset is only used to get an idea of wha could be results on evaluation dataset
    validation_dataset = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                    transform=transforms_valid)

    # ##############
    # Model
    # ##############

    if model_path is not None:
        state = torch.load(model_path)
        crnn_kwargs = state["model"]["kwargs"]
        crnn = CRNN(**crnn_kwargs)
        crnn.load(parameters=state["model"]["state_dict"])
    else:
        crnn_kwargs = {"n_in_channel": 1, "nclass": len(classes), "attention": True, "n_RNN_cell": 64,
                       "n_layers_RNN": 2,
                       "activation": cfg.activation,
                       "dropout": cfg.dropout, "kernel_size": 3 * [3], "padding": 3 * [1], "stride": 3 * [1],
                       "nb_filters": [64, 64, 64],
                       "pooling": list(3 * ((2, 4),))}
        crnn = CRNN(**crnn_kwargs)
        crnn.apply(weights_init)
    LOG.info(crnn)

    if model_path is None or resume:
        transforms = get_transforms(max_frames, scaler)
        train_weak_data.set_transform(transforms)
        train_synth_data.set_transform(transforms)

        concat_dataset = ConcatDataset([train_weak_data, train_synth_data])
        # Taking as much data from synthetic than strong.
        sampler = MultiStreamBatchSampler(concat_dataset,
                                          batch_sizes=[cfg.batch_size // 2, cfg.batch_size // 2])
        training_data = DataLoader(concat_dataset, batch_sampler=sampler)
        valid_synth_data = DataLoadDf(valid_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                      transform=transforms_valid)
        valid_weak_data = DataLoadDf(valid_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                     transform=transforms_valid)

        if model_path is not None and resume:
            optim_kwargs = state["optimizer"]["kwargs"]
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
            optimizer.load_state_dict(state["optimizer"]["state_dict"])
            starting_epoch = state["epoch"]
            LOG.info("Resuming at epoch: {}".format(starting_epoch))
        else:
            optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
            starting_epoch = 0
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
                          'state_dict': optimizer.state_dict()}
        }

        save_best_cb = SaveBest("sup")

        # ##############
        # Train
        # ##############
        # Eval 2018
        eval_2018_df = dataset.intialize_and_get_df(cfg.eval2018, reduced_number_of_data)
        eval_2018 = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                               transform=transforms_valid)

        [crnn] = to_cuda_if_available([crnn])
        for epoch in range(starting_epoch, cfg.n_epoch):
            crnn = crnn.train()

            train(training_data, crnn, optimizer, epoch)

            crnn = crnn.eval()
            train_predictions = get_predictions(crnn, train_synth_data, many_hot_encoder.decode_strong,
                                                save_predictions=None)
            train_predictions.onset = train_predictions.onset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
            train_predictions.offset = train_predictions.offset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
            train_metric = event_based_evaluation_df(train_synth_df, train_predictions)
            LOG.info(train_metric)

            weak_metric = get_f_measure_by_class(crnn, len(classes),
                                                 DataLoader(train_weak_data, batch_size=cfg.batch_size))
            LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
            LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

            predictions = get_predictions(crnn, valid_synth_data, many_hot_encoder.decode_strong,
                                          save_predictions=None)
            predictions.onset = predictions.onset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
            predictions.offset = predictions.offset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
            valid_metric = event_based_evaluation_df(valid_synth_df, predictions)

            # Eval 2018
            predictions = get_predictions(crnn, eval_2018, many_hot_encoder.decode_strong,
                                          save_predictions=None)
            predictions.onset = predictions.onset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
            predictions.offset = predictions.offset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
            eval2018_metric = event_based_evaluation_df(eval_2018_df, predictions)

            weak_metric = get_f_measure_by_class(crnn, len(classes),
                                                 DataLoader(valid_weak_data, batch_size=cfg.batch_size))

            LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
            LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))
            LOG.info(valid_metric)
            LOG.info("Eval_metric:")
            LOG.info(eval2018_metric)

            state['model']['state_dict'] = crnn.state_dict()
            state['optimizer']['state_dict'] = optimizer.state_dict()
            state['epoch'] = epoch
            state['valid_metric'] = valid_metric.results()
            if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
                model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
                torch.save(state, model_fname)

            if cfg.save_best:
                global_valid = valid_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
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
    transforms_valid = get_transforms(max_frames, scaler=scaler)
    # Eval 2018
    eval_2018_df = dataset.intialize_and_get_df(cfg.eval2018, reduced_number_of_data)
    eval_2018 = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                           transform=transforms_valid)
    predictions = get_predictions(crnn, eval_2018, many_hot_encoder.decode_strong,
                                  save_predictions=None)
    predictions.onset = predictions.onset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
    predictions.offset = predictions.offset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
    eval2018_metric = event_based_evaluation_df(eval_2018_df, predictions)
    eval2018_metric_seg = segment_based_evaluation_df(eval_2018_df, predictions)

    eval_2018_df.onset = eval_2018_df.onset * (cfg.sample_rate / cfg.hop_length) // pooling_time_ratio
    eval_2018_df.offset = eval_2018_df.offset * (cfg.sample_rate / cfg.hop_length) // pooling_time_ratio
    validation_dataset = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                    transform=transforms_valid)
    weak_metric = get_f_measure_by_class(crnn, len(classes), DataLoader(eval_2018, batch_size=cfg.batch_size))
    LOG.info("2018: ")
    LOG.info(eval2018_metric)
    LOG.info(eval2018_metric_seg)
    LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    # Validation 2019
    validation_dataset = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                    transform=transforms_valid)
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.csv")
    predictions = get_predictions(crnn, validation_dataset, many_hot_encoder.decode_strong,
                                  save_predictions=predicitons_fname)
    predictions.onset = predictions.onset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
    predictions.offset = predictions.offset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
    metric = event_based_evaluation_df(validation_df, predictions)
    metric_seg = segment_based_evaluation_df(validation_df, predictions)

    validation_df.onset = validation_df.onset * (cfg.sample_rate / cfg.hop_length) // pooling_time_ratio
    validation_df.offset = validation_df.offset * (cfg.sample_rate / cfg.hop_length) // pooling_time_ratio
    validation_dataset = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                    transform=transforms_valid)
    weak_metric = get_f_measure_by_class(crnn, len(classes),
                                         DataLoader(validation_dataset, batch_size=cfg.batch_size))
    LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))
    LOG.info("FINAL predictions")
    LOG.info(metric)
    LOG.info(metric_seg)

