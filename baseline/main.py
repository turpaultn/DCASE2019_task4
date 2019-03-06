import argparse
import os
import pandas as pd
import time
import numpy as np

import scipy
import torch
from dcase_util.data import ProbabilityEncoder
from torch.utils.data import DataLoader

from DatasetDcase2019Task4 import DatasetDcase2019Task4
from DataLoad import DataLoadDf, ConcatDataset, ApplyLog, PadOrTrunc, ToTensor, Normalize, Compose, \
    MultiStreamBatchSampler, AugmentGaussianNoise
from Scaler import Scaler
from evaluation_measures import event_based_evaluation_df, get_f_measure_by_class
from models.CRNN import CRNN
import config as cfg
from utils import ManyHotEncoder, AverageMeterSet, create_folder, SaveBest, to_cuda_if_available
import ramps
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


def train(train_loader, model, ema_model, optimizer, epoch, global_step):
    class_criterion = nn.BCELoss()
    consistency_criterion_strong = nn.MSELoss()
    consistency_criterion_weak = nn.MSELoss()
    [class_criterion, consistency_criterion_weak, consistency_criterion_strong] = to_cuda_if_available(
        [class_criterion, consistency_criterion_weak, consistency_criterion_strong])

    meters = AverageMeterSet()

    start = time.time()
    for i, (input, ema_input, target) in enumerate(train_loader):
        e_step = epoch + i / len(train_loader)
        if global_step < cfg.rampup_length:
            rampup_value = ramps.sigmoid_rampup(e_step, cfg.rampup_length)
        else:
            rampup_value = 1.0

        if global_step > (cfg.train_iter_count - cfg.rampdown_length):
            rampdown_value = ramps.sigmoid_rampdown(e_step, cfg.rampdown_length)
        else:
            rampdown_value = 1.0

        adjust_learning_rate(optimizer, rampup_value, rampdown_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        labeled_minibatch_size = target.data.ne(-1).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        [input, ema_input, target] = to_cuda_if_available([input, ema_input, target])

        # Outputs
        ema_model_out = ema_model(ema_input)
        ema_model_out = ema_model_out.detach()
        strong_pred_ema = ema_model_out
        weak_pred_ema = torch.mean(ema_model_out, 1)

        model_out = model(input)
        strong_pred = model_out
        weak_pred = torch.mean(model_out, 1)

        # Weak BCE Loss
        # Trick to not take unlabeled data
        weak_mask = target.mean(-1).mean(-1) != -1
        target_weak = target.max(-2)[0]
        weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
        if i < 3:
            LOG.debug("target: {}".format(target.mean(-2)))
            LOG.debug("mask: {}".format(weak_mask))
            LOG.debug("Target_weak: {}".format(target_weak))
            LOG.debug("Target_weak mask: {}".format(target_weak[weak_mask]))
            LOG.debug(weak_class_loss)
        meters.update('weak_class_loss', weak_class_loss.item())

        ema_class_loss = class_criterion(weak_pred_ema[weak_mask], target_weak[weak_mask])
        meters.update('weak_ema_class_loss', ema_class_loss.item())

        loss = weak_class_loss

        # Strong BCE loss
        strong_size = train_loader.batch_sampler.batch_sizes[-1]
        strong_class_loss = class_criterion(strong_pred[-strong_size:], target[-strong_size:])
        meters.update('strong_class_loss', strong_class_loss.item())

        strong_ema_class_loss = class_criterion(strong_pred_ema[-strong_size:], target[-strong_size:])
        meters.update('strong_ema_class_loss', strong_ema_class_loss.item())

        loss += strong_class_loss

        # Consistency losses
        consistency_cost = cfg.max_consistency_cost * rampup_value
        if cfg.consistency_weak:
            meters.update('cons_weight_weak', consistency_cost)
            consistency_loss_weak = consistency_cost * consistency_criterion_weak(weak_pred, weak_pred_ema)
            meters.update('weak_cons_loss', consistency_loss_weak.item())
            loss += consistency_loss_weak

        if cfg.consistency_strong:
            meters.update('cons_weight', consistency_cost)
            consistency_loss_strong = consistency_cost * consistency_criterion_strong(strong_pred, strong_pred_ema)
            meters.update('strong_cons_loss', consistency_loss_strong.item())
            loss += consistency_loss_strong

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, 0.999, global_step)

    meters.update('epoch_time', time.time() - start)

    LOG.info(
        'Epoch: {}\t'
        'Time {meters[epoch_time]:.2f}\t'
        'LR {meters[lr]:.2E}\t'
        'Weak_loss {meters[weak_class_loss]:.4f}\t'
        'Strong_loss {meters[strong_class_loss]:.4f}\t'
        'Weak Cons {meters[weak_cons_loss]:.4f}\t'
        'Srtong Cons {meters[strong_cons_loss]:.4f}\t'
        'EMA loss {meters[weak_ema_class_loss]:.4f}\t'
        'Strong EMA loss {meters[strong_ema_class_loss]:.4f}\t'.format(
            epoch, meters=meters))
    return global_step


def get_predictions(model, valid_dataset, decoder, in_seconds=True, save_predictions=None):
    for i, (input, _) in enumerate(valid_dataset):
        [input] = to_cuda_if_available([input])

        pred_strong = model(input.unsqueeze(0))
        pred_strong = pred_strong.cpu()
        pred_strong = pred_strong.squeeze(0).detach().numpy()
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)

        pred_strong = scipy.ndimage.filters.median_filter(pred_strong, (cfg.median_window, 1))
        pred = decoder(pred_strong)
        pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
        pred["filename"] = valid_dataset.df.iloc[i]["filename"]
        if i == 0:
            LOG.debug(pred)
            LOG.debug(pred_strong)
            prediction_df = pred.copy()
        else:
            prediction_df = prediction_df.append(pred)

    if in_seconds:
        prediction_df.onset = prediction_df.onset / (cfg.sample_rate / cfg.hop_length)
        prediction_df.offset = prediction_df.offset / (cfg.sample_rate / cfg.hop_length)
    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions)
    return prediction_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--subpart_data', type=int, default=None, dest="subpart_data")
    f_args = parser.parse_args()

    reduced_number_of_data = f_args.subpart_data
    LOG.info("subpart_data = " + str(reduced_number_of_data))
    max_frames = cfg.max_frames

    saved_model_dir = "stored_data/model"
    saved_pred_dir = "stored_data/predictions"
    create_folder(saved_model_dir)
    create_folder(saved_pred_dir)

    # ##############
    # DATA
    # ##############
    dataset = DatasetDcase2019Task4(os.path.join(".."),
                                    base_feature_dir=os.path.join("..", "dataset", "features"),
                                    subpart_data=reduced_number_of_data,
                                    save_log_feature=False)

    def get_transforms(frames, scaler=None, add_axis_conv=True, augment_type=None):
        transf = []
        unsqueeze_axis = None
        if add_axis_conv:
            unsqueeze_axis = 0

        # Todo, add other augmentations
        if augment_type is not None:
            if augment_type == "noise":
                transf.append(AugmentGaussianNoise(mean=0., std=0.5))

        transf.extend([ApplyLog(), PadOrTrunc(nb_frames=frames), ToTensor(unsqueeze_axis=unsqueeze_axis)])
        if scaler is not None:
            transf.append(Normalize(scaler=scaler))

        return Compose(transf)

    weak_df = dataset.intialize_and_get_df(cfg.weak, reduced_number_of_data)
    unlabel_df = dataset.intialize_and_get_df(cfg.unlabel, reduced_number_of_data)
    synthetic_df = dataset.intialize_and_get_df(cfg.synthetic, reduced_number_of_data, download=False)
    validation_df = dataset.intialize_and_get_df(cfg.validation, reduced_number_of_data)

    classes = DatasetDcase2019Task4.get_classes([weak_df, validation_df, synthetic_df])
    many_hot_encoder = ManyHotEncoder(classes, n_frames=max_frames)

    transforms = get_transforms(max_frames)

    train_weak_df = weak_df.sample(frac=0.8, random_state=10).reset_index(drop=True)
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)

    train_synth_df = synthetic_df.sample(frac=0.8, random_state=10).reset_index(drop=True)
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)

    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_length
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_length

    train_weak_data = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                 transform=transforms)
    unlabel_data = DataLoadDf(unlabel_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                              transform=transforms)
    train_synth_data = DataLoadDf(train_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms)

    scaler = Scaler()
    scaler.calculate_scaler(ConcatDataset([train_weak_data, unlabel_data]))
    LOG.debug(scaler.mean_)

    transforms = get_transforms(max_frames, scaler, augment_type="noise")
    train_weak_data.set_transform(transforms)
    unlabel_data.set_transform(transforms)
    train_synth_data.set_transform(transforms)

    concat_dataset = ConcatDataset([train_weak_data, unlabel_data, train_synth_data])
    sampler = MultiStreamBatchSampler(concat_dataset,
                                      batch_sizes=[cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4])
    training_data = DataLoader(concat_dataset, batch_sampler=sampler)

    transforms_valid = get_transforms(max_frames, scaler=scaler)
    valid_synth_data = DataLoadDf(valid_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms_valid)
    valid_weak_data = DataLoadDf(valid_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms_valid)
    validation_dataset = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                    transform=transforms_valid)

    # ##############
    # Model
    # ##############
    crnn_kwargs = {"n_in_channel": 1, "nclass":len(classes), "activation": "Relu",
                   "mode": "both", "dropout": cfg.conv_dropout, "max_frames": max_frames}
    crnn = CRNN(**crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

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
                      'state_dict': optimizer.state_dict()}
    }

    save_best_cb = SaveBest("sup")

    # ##############
    # Train
    # ##############
    global_step = 0
    epoch = 0
    while global_step < cfg.train_iter_count:
        crnn = crnn.train()
        crnn_ema = crnn_ema.train()

        [crnn, crnn_ema] = to_cuda_if_available([crnn, crnn_ema])

        global_step = train(training_data, crnn, crnn_ema, optimizer, epoch, global_step)

        crnn = crnn.eval()
        predictions = get_predictions(crnn, valid_synth_data, many_hot_encoder.decode_strong,
                                      save_predictions=None)

        valid_metric = event_based_evaluation_df(valid_synth_df, predictions)
        weak_metric = get_f_measure_by_class(crnn, len(classes),
                                             DataLoader(valid_weak_data, batch_size=cfg.batch_size))

        LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
        LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))
        LOG.info(valid_metric)

        state['model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
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
        epoch += 1

    if cfg.save_best:
        state = torch.load(os.path.join(saved_model_dir, "baseline_best"))
        crnn.load(parameters=state["model"]["state_dict"])

    # ##############
    # Validation
    # ##############
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.csv")
    predictions = get_predictions(crnn, validation_dataset, many_hot_encoder.decode_strong,
                                  save_predictions=predicitons_fname)
    metric = event_based_evaluation_df(validation_df, predictions)
    LOG.info("FINAL predictions")
    LOG.info(metric)
