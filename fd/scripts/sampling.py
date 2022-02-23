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

from fd.sampling import sample_patch, SamplingConfig, prepare_eopatches_paths
from fd.utils import multiprocess, LogFileFilter

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def sample_patchlets(config: dict):
    """ Call utilities to sample patchlets from eopatches

    Args:
        config (dict): Configuration dictionary with sampling parameters
    """
    sampling_config = SamplingConfig(
        bucket_name=config['bucket_name'], 
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'],
        eopatches_location=config['eopatches_location'],
        output_path=config['output_path'],
        sample_positive=config['sample_positive'],
        mask_feature_name=config['mask_feature_name'],
        buffer=config['buffer'],
        patch_size=config['patch_size'],
        num_samples=config['num_samples'],
        max_retries=config['max_retries'],
        fraction_valid=config['fraction_valid'],
        sampled_feature_name=config['sampled_feature_name'],
        cloud_coverage=config['cloud_coverage']
    )

    eopatches_paths = prepare_eopatches_paths(sampling_config)

    process_fn = partial(sample_patch, sampling_config=sampling_config)

    LOGGER.info('Running sampling of patchlets')
    _ = multiprocess(process_fun=process_fn, arguments=eopatches_paths, max_workers=config['max_workers'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample patchelts from EOPatches.\n")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file with sampling parameters",
        required=True
    )
    args = parser.parse_args()

    LOGGER.info(f'Reading configuration from {args.config}')
    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    sample_patchlets(cfg_dict)
