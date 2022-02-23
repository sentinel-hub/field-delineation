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

from fd.vectorisation import (
    VectorisationConfig,
    run_vectorisation
)
from fd.utils import LogFileFilter


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def vectorise(config: dict):
    """ Utility function to run vectorisation of post-processed pseudo-probas

    Args:
        config (dict): configuration parameters for vectorisation
    """
    vectorisation_config = VectorisationConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config['aws_access_key_id'], 
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'],
        tiffs_folder=config['tiffs_folder'],
        time_intervals=config['time_intervals'],
        utms=config['utms'],
        shape=tuple(config['shape']),
        buffer=tuple(config['buffer']),
        weights_file=config['weights_file'],
        vrt_dir=config['vrt_dir'],
        predictions_dir=config['predictions_dir'],
        contours_dir=config['contours_dir'],
        max_workers=config['max_workers'],
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        threshold=config['threshold'],
        cleanup=config['cleanup'],
        skip_existing=config['skip_existing'],
        rows_merging=config['rows_merging']
    )
    LOGGER.info('Running vectorisation')
    list_of_merged_files = run_vectorisation(vectorisation_config)

    LOGGER.info(f'Vectorised files created are {list_of_merged_files}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run vectorisation of predicted pseudo-probability maps.\n")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file with parameters required for vectorisation", 
        required=True
    )
    args = parser.parse_args()

    LOGGER.info(f'Reading configuration from {args.config}')
    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    vectorise(cfg_dict)
