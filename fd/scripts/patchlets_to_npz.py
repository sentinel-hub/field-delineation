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
import sys
import json
import logging
import argparse

from functools import partial 

from fd.utils import prepare_filesystem, multiprocess, LogFileFilter
from fd.create_npz_files import (
    CreateNpzConfig, 
    extract_npys, 
    concatenate_npys, 
    save_into_chunks
)


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)



def patchlets_to_npz_files(config: dict):
    """ Utility function to convert patchlets to npz files

    Args:
        config (dict): Configuration dictionary with conversion parameters
    """

    npz_config = CreateNpzConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'], 
        patchlets_folder=config['patchlets_folder'],
        output_folder=config['output_folder'], 
        output_dataframe=config['output_dataframe'],
        chunk_size=config['chunk_size'])

    filesystem = prepare_filesystem(npz_config)
    
    LOGGER.info('Read patchlet names from bucket')
    patchlets = [os.path.join(npz_config.patchlets_folder, eop_name)
                for eop_name in filesystem.listdir(npz_config.patchlets_folder)]

    partial_fn = partial(extract_npys, cfg=npz_config)

    LOGGER.info('Collect npy patchlet files')
    npys = multiprocess(partial_fn, patchlets, max_workers=config['max_workers'])

    npys_dict = concatenate_npys(npys)

    LOGGER.info('Save files into npz chunks')
    save_into_chunks(npz_config, npys_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert patchlets to npz files.\n")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file with npz conversion parameters", 
        required=True
    )
    args = parser.parse_args()
    LOGGER.info(f'Reading configuration from {args.config}')
    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    patchlets_to_npz_files(cfg_dict)
