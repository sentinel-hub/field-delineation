#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import sys
import json
import logging
import argparse
from functools import partial

import numpy as np
import pandas as pd

from fd.utils import multiprocess, prepare_filesystem, LogFileFilter
from fd.training import SplitConfig, fold_split


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def k_fold_split(config: dict):
    """ Split dataset of patchlets into k-folds

    Args:
        config (dict): Config dictionary with k-fold splitting params
    """

    split_config = SplitConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config['aws_access_key_id'], 
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'],
        metadata_path=config['metadata_path'],
        npz_folder=config['npz_folder'],
        n_folds=config['n_folds'],
        seed=config['seed']
    )

    filesystem = prepare_filesystem(split_config)

    LOGGER.info(f'Read metadata file {split_config.metadata_path}')
    with filesystem.open(split_config.metadata_path, 'rb') as fcsv:     
        df = pd.read_csv(fcsv)      

    eops = df.eopatch.unique()

    LOGGER.info('Assign folds to eopatches')
    np.random.seed(seed=split_config.seed)
    fold = np.random.randint(1, high=split_config.n_folds+1, size=len(eops))
    eopatch_to_fold_map = dict(zip(eops, fold))

    df['fold'] = df['eopatch'].apply(lambda x: eopatch_to_fold_map[x])

    for nf in range(split_config.n_folds):
        LOGGER.info(f'{len(df[df.fold==nf+1])} patchlets in fold {nf+1}')

    LOGGER.info('Split files into folds')
    partial_fn = partial(fold_split, df=df, config=split_config)

    npz_files = filesystem.listdir(split_config.npz_folder)

    npz_files = [npzf for npzf in npz_files if npzf.startswith('patchlets_')]

    _ = multiprocess(partial_fn, npz_files, max_workers=config['max_workers'])

    LOGGER.info('Update metadata file with fold information')
    with filesystem.open(split_config.metadata_path, 'w') as fcsv:
        df.to_csv(fcsv, index=False)


if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description="Split patchlet dataset into k-folds.\n")
    parser.add_argument(
    "--config", 
    type=str, 
    help="Path to config file with parameters required for k-fold splitting", 
    required=True
    )
    args = parser.parse_args()
    LOGGER.info(f'Reading configuration from {args.config}')

    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    k_fold_split(cfg_dict)
