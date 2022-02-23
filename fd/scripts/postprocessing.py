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
import warnings
import argparse

import geopandas as gpd

from eolearn.core import FeatureType, EOExecutor

from fd.post_processing import (
    get_post_processing_workflow, 
    get_exec_args, 
    PostProcessConfig)
from fd.utils import LogFileFilter


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def run_post_processing(config: dict):
    """ Utility function to run post-processing on predictions

    Args:
        config (dict): Dictionary with config parameters for post-processing
    """

    post_process_config = PostProcessConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config['aws_access_key_id'], 
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'],
        time_intervals=config['time_intervals'],
        eopatches_folder=config['eopatches_folder'],
        tiffs_folder=config['tiffs_folder'],
        feature_extent=(FeatureType(config['feature_extent'][0]), config['feature_extent'][1]),
        feature_boundary=(FeatureType(config['feature_boundary'][0]), config['feature_boundary'][1]),
        model_version=config['model_version'],
        max_cloud_coverage=config['max_cloud_coverage'],
        percentile=config['percentile'],
        scale_factor=config['scale_factor'],
        disk_size=config['disk_size']
    )
    LOGGER.info('Reading grid definition')
    grid_definition = gpd.read_file(config['grid_filename'])

    eopatches_list = grid_definition.name.values

    workflow = get_post_processing_workflow(post_process_config)

    LOGGER.info('Prepare arguments for execution')
    exec_args = get_exec_args(workflow=workflow, 
                              eopatch_list=eopatches_list,
                              config=post_process_config)

    warnings.simplefilter(action='ignore', category=UserWarning)
    LOGGER.info('Run execution')
    executor = EOExecutor(workflow, exec_args, save_logs=True)
    executor.run(workers=config['max_workers'])

    LOGGER.info('Making execution report')
    executor.make_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply post-processing to the predicted pseudo-probabilities.\n")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file with parameters required for post-processing", 
        required=True
    )
    args = parser.parse_args()
    LOGGER.info(f'Reading configuration from {args.config}')

    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    run_post_processing(cfg_dict)
