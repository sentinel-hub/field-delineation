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
from typing import List, Tuple
from functools import partial

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

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
