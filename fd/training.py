#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import os
import json
from datetime import datetime
from typing import List, Tuple, Callable
from functools import partial

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from wandb.keras import WandbCallback

from eoflow.models.metrics import MCCMetric
from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.losses import JaccardDistanceLoss, TanimotoDistanceLoss
from eoflow.models.segmentation_unets import ResUnetA

from .utils import BaseConfig, prepare_filesystem
from .tf_data_utils import (
    npz_dir_dataset,
    normalize_meanstd, normalize_perc,
    Unpack, ToFloat32, augment_data, FillNaN, OneMinusEncoding, LabelsToDict)


@dataclass
class SplitConfig(BaseConfig):
    metadata_path: str
    npz_folder: str
    n_folds: int
    seed: int = 42


@dataclass
class TrainingConfig(BaseConfig):
    input_shape: Tuple[int, int, int]
    n_classes: int
    batch_size: int
    iterations_per_epoch: int
    num_epochs: int
    npz_folder: str
    metadata_path: str
    model_folder: str
    model_s3_folder: str
    model_name: str
    augmentations_feature: List[str]
    augmentations_label: List[str]
    reference_names: List[str]
    normalize: str
    model_config: dict
    chkpt_folder: str = None
    n_folds: int = None
    wandb_id: str = None
    fill_value: int = -2
    seed: int = 42
    wandb_project: str = 'field-delineation'


NORMALIZER = dict(to_meanstd=partial(normalize_meanstd, subtract='mean'),
                  to_medianstd=partial(normalize_meanstd, subtract='median'),
                  to_perc=normalize_perc)


def fold_split(chunk: str, df: pd.DataFrame, config: SplitConfig) -> None:
    """ Extract from chunk file patchlets to each fold """

    filesystem = prepare_filesystem(config)

    data = np.load(filesystem.openbin(os.path.join(config.npz_folder, chunk)), allow_pickle=True)

    for fold in range(1, config.n_folds + 1):

        idx_fold = df[(df.chunk == chunk) & (df.fold == fold)].chunk_pos

        if not idx_fold.empty:
            patchlets = {}
            for key in data:
                patchlets[key] = data[key][idx_fold]

            fold_folder = os.path.join(config.npz_folder, f'fold_{fold}')

            filesystem.makedir(fold_folder, recreate=True)

            np.savez(filesystem.openbin(os.path.join(fold_folder, chunk), 'wb'), **patchlets)

            del idx_fold, patchlets


def get_dataset(config: TrainingConfig, fold: int, augment: bool, num_parallel: int,
                randomize: bool = True, npz_from_s3: bool = False) -> tf.data.Dataset:
    """ Get TF Dataset for a given fold, loading npz files from directory

    :param config: configuration dictionary for pipeline
    :param fold: which fold directory to load from
    :param augment: whether to augment the data or not
    :param num_parallel: number of npz files to interleave when creating the dataset
    :param randomize: whether to shuffle the sample order in the dataset
    :param npz_from_s3: whether to load npz files from local disk or from an S3 bucket
    """
    assert config.normalize in ['to_meanstd', 'to_medianstd', 'to_perc']

    filesystem = prepare_filesystem(config)

    data = dict(X='features', y_extent='y_extent', y_boundary='y_boundary', y_distance='y_distance')

    dataset = npz_dir_dataset(os.path.join(config.npz_folder, f'fold_{fold}'), data, metadata_path=config.metadata_path,
                              fold=fold, randomize=randomize, num_parallel=num_parallel, filesystem=filesystem, npz_from_s3=npz_from_s3)

    normalizer = NORMALIZER[config.normalize]

    augmentations = [augment_data(config.augmentations_feature, config.augmentations_label)] if augment else []
    dataset_ops = [normalizer, Unpack(), ToFloat32()] + augmentations + [FillNaN(fill_value=config.fill_value),
                                                                         OneMinusEncoding(n_classes=config.n_classes),
                                                                         LabelsToDict(config.reference_names)]

    for dataset_op in dataset_ops:
        dataset = dataset.map(dataset_op)

    return dataset


def initialise_model(config: TrainingConfig, chkpt_folder: str = None):
    """ Initialise ResUnetA model 
    
    If an existing chekpoints directory is provided, the existing weights are loaded and 
    training starts from existing state
    """
    # TODO: This metric was removed as it breaks training. Meed to investigate and fix 
    # mcc_metric = MCCMetric(default_n_classes=config.n_classes, default_threshold=.5)
    # mcc_metric.init_from_config({'n_classes': config.n_classes})
    model = ResUnetA(config.model_config)
    
    model.build(dict(features=[None] + list(config.input_shape)))
    
    model.net.compile(
        loss={'extent':TanimotoDistanceLoss(from_logits=False),
              'boundary':TanimotoDistanceLoss(from_logits=False),
              'distance':TanimotoDistanceLoss(from_logits=False)},
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.model_config['learning_rate']),
        # comment out the metrics you don't care about
        metrics=[segmentation_metrics['accuracy'](),
                 tf.keras.metrics.MeanIoU(num_classes=config.n_classes)])
    
    if chkpt_folder is not None:
        model.net.load_weights(f'{chkpt_folder}/model.ckpt')
        
    return model


def initialise_callbacks(config: TrainingConfig, 
                         fold: int) -> Tuple[str, List[Callable]]:
    """ Initialise callbacks used for logging and saving of models """
    now = datetime.now().isoformat(sep='-', timespec='seconds').replace(':', '-')
    model_path = f'{config.model_folder}/{config.model_name}_fold-{fold}_{now}'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logs_path = os.path.join(model_path, 'logs')
    checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')


    # Tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path,
                                                          update_freq='epoch',
                                                          profile_batch=0)

    # Checkpoint saving callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path,
                                                             save_best_only=True,
                                                             save_freq='epoch',
                                                             save_weights_only=True)

    full_config = dict(**config.model_config, 
                       iterations_per_epoch=config.iterations_per_epoch, 
                       num_epochs=config.num_epochs, 
                       batch_size=config.batch_size,
                       model_name=f'{config.model_name}_{now}'
                      )

    # Save model config 
    with open(f'{model_path}/model_cfg.json', 'w') as jfile:
        json.dump(config.model_config, jfile)

    # initialise wandb if used
    if config.wandb_id:
        wandb.init(config=full_config, 
                   name=f'{config.model_name}-leftoutfold-{fold}',
                   project=config.wandb_project, 
                   sync_tensorboard=True)
        
    callbacks = [tensorboard_callback, 
                 checkpoint_callback, 
#                  visualisation_callback
                ] + ([WandbCallback()] if config.wandb_id is not None else [])
    
    return model_path, callbacks 
