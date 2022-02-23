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

from fd.compute_normalization import (ComputeNormalizationConfig, 
                                      stats_per_npz_ts, 
                                      prepare_filesystem,
                                      concat_npz_results,
                                      create_per_band_norm_dataframe)
from fd.utils import multiprocess, LogFileFilter


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def calculate_normalization_factors(config: dict):
    """ Utility function to calculate normalisation factors from the npz files

    Args:
        config (dict): Configuration parameters for normalization factors calculation
    """

    norm_config = ComputeNormalizationConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config['aws_access_key_id'], 
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'],
        npz_files_folder=config['npz_files_folder'],
        metadata_file=config['metadata_file']
        )

    filesystem = prepare_filesystem(norm_config)

    npz_files = filesystem.listdir(norm_config.npz_files_folder)

    LOGGER.info('Compute stats per patchlet')
    partial_fn = partial(stats_per_npz_ts, config=norm_config)
    results = multiprocess(partial_fn, npz_files, max_workers=config['max_workers'])

    stats_keys = ['mean', 'std', 'median', 'perc_99']
    identifier_keys = ['timestamp', 'patchlet'] 

    concatenated_stats = {}

    for key in stats_keys+identifier_keys: 
        concatenated_stats[key] = concat_npz_results(key, results)

    LOGGER.info('Create dataframe with normalisation factors')
    df = create_per_band_norm_dataframe(concatenated_stats, stats_keys, identifier_keys)

    # convert to datetime
    timestamps = df['timestamp'].apply(lambda d: d.tz_localize(None))
    df['timestamp']=timestamps.astype(np.datetime64)

    # add "month" period
    df['month']=df.timestamp.dt.to_period("M")

    norm_cols = [norm.format(band) 
                for norm in ['perc_99_b{0}_mean', 
                            'mean_b{0}_mean', 
                            'median_b{0}_mean', 
                            'std_b{0}_mean'] for band in range(4)]

    aggs = {}
    stat_cols = []
    stats = ['perc_99', 'mean', 'median', 'std']
    bands = list(range(4))
    for stat in stats:
        for band in bands:
            aggs[f'{stat}_b{band}'] = [np.std, np.mean]
            stat_cols.append(f'{stat}_b{band}')

    LOGGER.info('Aggregate normalization stats by month')
    monthly = pd.DataFrame(df.groupby('month', as_index=False)[stat_cols].agg(aggs))
    monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
    monthly.rename(columns={'month_':'month'}, inplace=True)

    def norms(month):
        return monthly.loc[monthly.month==month][norm_cols].values[0]

    df['norm_perc99_b0'], df['norm_perc99_b1'], df['norm_perc99_b2'], df['norm_perc99_b3'], \
    df['norm_meanstd_mean_b0'], df['norm_meanstd_mean_b1'], df['norm_meanstd_mean_b2'], df['norm_meanstd_mean_b3'], \
    df['norm_meanstd_median_b0'], df['norm_meanstd_median_b1'], df['norm_meanstd_median_b2'], df['norm_meanstd_median_b3'], \
    df['norm_meanstd_std_b0'], df['norm_meanstd_std_b1'], df['norm_meanstd_std_b2'], df['norm_meanstd_std_b3'] = zip(*map(norms, df.month))

    LOGGER.info(f'Read metadata file {norm_config.metadata_file}')
    with filesystem.open(norm_config.metadata_file, 'rb') as fcsv:
        df_info = pd.read_csv(fcsv)

    df_info['timestamp'] = pd.to_datetime(df_info.timestamp)

    timestamps = df_info['timestamp'].apply(lambda d: d.tz_localize(None))
    df_info['timestamp'] = timestamps.astype(np.datetime64)

    LOGGER.info('Add normalization information to metadata file')
    new_df = df_info.merge(df, how='inner', on=['patchlet', 'timestamp'])

    LOGGER.info('Overwrite metadata file with new file')
    with filesystem.open(norm_config.metadata_file, 'w') as fcsv:
        new_df.to_csv(fcsv, index=False)


if __name__ == '__main__':
     
    parser = argparse.ArgumentParser(description="Calculate normalization factors.\n")

    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file with parameters required for calculation on normalization factors", 
        required=True
    )
    args = parser.parse_args()

    LOGGER.info(f'Reading configuration from {args.config}')
    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    calculate_normalization_factors(cfg_dict)

