import math
import torch
from torch import nn, optim
from aux_files.data_loading import AllTweets2021Dataset

default_args = {
        # script
        'data_dir': './data',
        'arch': 'char_lstm',
        'run_name': None,  # TODO: ???
        'save_prefix': './conf_geo_model',
        'subsample': 9e9,
        # data
        'split_uids': True,  # Hold-out train user IDs when generating validation
        'max_seq_len': -1,  # Truncate tweet text to length; -1 to disable
        'truncate_ratio': False,  # Subsample dataset to this ratio
        'dataset_name': 'all_tweeets_2021_dataset',
        # training
        'loss': 'mse',
        'optimizer': 'adamw',
        'metric': 'gc_distance',
        'gradient_accumulation_steps': 8,  # Number of training steps to accumulate gradient over
        'scheduler': 'constant',  # `linear' scales up the LR over `args.warmup_ratio' steps
        'warmup_ratio': 0.2,  # Ratio of maximum training steps to scale LR over
        'lr': 1e-4,  # Optimiser learning rate
        'dropout': 0.3,  # Model dropout ratio
        'batch_size': 128,  #Training batch size
        'num_epochs': 10,  # Number of training epochs
        'reduce_layer': False,  # Add linear layer before output
        'conf_estim': False,  # Use confidence estimation
        'conf_p': 0.6,  # Estimation hyperparam; top-p% = 1
        'conf_lambda': 0.1,  # Confidence loss weight
        'confidence_bands': [2, 5, 10, 25, 50, 100],
        'model_save_band': 25,  # Save model based on performance in band
        'no_conf_epochs': 0.2,  # Proportion of total no. of epochs after which to start co-training confidence estimator")
        'confidence_validation_criterion': False,  # Save models based on the score on the highest confidence level instead of on all data")
    }

def get_dataset(dataset_name):
    datasets = {
        'all_tweeets_2021_dataset': AllTweets2021Dataset,
    }
    return datasets[dataset_name]


def get_criterion(crit):
    crits = {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss(),
        'cross_entropy': nn.CrossEntropyLoss(),
    }
    return crits[crit]


def get_optimizer(opt):
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
    }
    return optimizers[opt]

def get_metric(metric):
    metrics = {
        'gc_distance': gc_distance
    }
    return metrics[metric]


def nan_to_num_(x):
    # for torch version less than 1.8
    x[torch.isnan(x)] = 0.0000e+00
    x[torch.isinf(x) & (x > 0)] = 3.4028e+38
    x[torch.isinf(x) & (x < 0)] = -3.4028e+38
    return x

def gc_distance(gold, pred):
    EARTH_RADIUS = 6372.8
    _degree_radian = lambda d: (d * math.pi) / 180
    rad_gold = _degree_radian(gold)
    rad_pred = _degree_radian(pred)

    cos_gold = torch.cos(rad_gold)
    sin_gold = torch.sin(rad_gold)
    cos_pred = torch.cos(rad_pred)
    sin_pred = torch.sin(rad_pred)

    n_gold = torch.stack([cos_gold[:, 1] * cos_gold[:, 0], cos_gold[:, 1] * sin_gold[:, 0], sin_gold[:, 1]], dim=1)
    n_pred = torch.stack([cos_pred[:, 1] * cos_pred[:, 0], cos_pred[:, 1] * sin_pred[:, 0], sin_pred[:, 1]], dim=1)
    
    return nan_to_num_(torch.acos(torch.mm(n_gold.to(gold.device), n_pred.to(pred.device).T).diag()) * EARTH_RADIUS)
    #return torch.nan_to_num(torch.acos(torch.inner(n_gold.to(gold.device), n_pred.to(pred.device)).diag()) * EARTH_RADIUS)  # for torch 1.8^