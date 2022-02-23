#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import argparse
import json
import logging
import sys
import time

from fd.download import (
    DownloadConfig,
    create_batch_request,
    monitor_batch_job,
    get_tile_status_counts, get_batch_tiles
)
from sentinelhub import (
    BatchSplitter,
    DataCollection,
    MimeType,
    SentinelHubRequest, SentinelHubBatch, BatchRequestStatus
)
from fd.utils import LogFileFilter

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def get_user_input(prompt: str) -> str:
    """ Get user answer given prompt

    Args:
        prompt (str): String with to confirm action by user

    Returns:
        str: "y" or "n" strings 
    """
    response = input(prompt)
    assert response in ['y', 'n'], 'Invalid input entered by user'
    return response


def batch_download(config: dict):
    """ Utility function to download S2 data using Batch Process API

    Args:
        config (dict): Configuration dictionary with 
    """
    download_config = DownloadConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config.get('aws_access_key_id'),
        aws_secret_access_key=config.get('aws_secret_access_key'),
        aws_region=config['aws_region'],
        sh_client_id=config['sh_client_id'],
        sh_client_secret=config['sh_client_secret'],
        aoi_filename=config['aoi_filename'],
        time_interval=config['time_interval'],
        data_collection=DataCollection.SENTINEL2_L1C,
        grid_definition=config['grid_definition'],
        tiles_path=config['tiles_path'],
        maxcc=config['maxcc'],
        mosaicking_order=config['mosaicking_order']
    )

    output_responses = [SentinelHubRequest.output_response(band, MimeType.TIFF) for band in config['bands']] + \
                       [SentinelHubRequest.output_response('userdata', MimeType.JSON)]

    LOGGER.info('Creating SH Batch request')
    batch = SentinelHubBatch()
    batch_request = create_batch_request(
        batch=batch,
        config=download_config,
        output_responses=output_responses,
        description=config['description']
    )
    LOGGER.info(batch_request)

    response = get_user_input('Want to start analysis of batch request? y/n: ')
    if response == 'n':
        return

    LOGGER.info('Starting analysis of batch request ...')
    batch.start_analysis(batch_request)

    batch_request = batch.get_request(batch_request)
    while batch_request.status != BatchRequestStatus.ANALYSIS_DONE:
        time.sleep(15)
        batch_request = batch.get_request(batch_request)
    LOGGER.info(batch_request)

    LOGGER.info('Creating splitter')
    splitter = BatchSplitter(batch_request=batch_request)
    grid_gdf = get_batch_tiles(splitter)

    LOGGER.info(f'Writing grid definition file to {config["grid_filename"]}')
    grid_gdf.to_file(config['grid_filename'], driver='GPKG')

    response = get_user_input('Want to start running the batch request? This action will use your PUs: y/n: ')
    if response == 'n':
        return

    batch.start_job(batch_request)

    batch_request = batch.get_request(batch_request)

    monitor_batch_job(batch, batch_request)

    LOGGER.info(f'Status of batch run is: {get_tile_status_counts(batch, batch_request)}')

    response = get_user_input('Did some tiles fail and you want to restart the process? y/n: ')
    if response == 'y':
        batch.restart_job(batch_request)

    LOGGER.info('Data download complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download S2 L1C data using Sentinel Hub Batch Process API.\n")
    parser.add_argument("--config", type=str, help="Path to config file with data download parameters", required=True)
    args = parser.parse_args()
    LOGGER.info(f'Reading configuration from {args.config}')
    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    batch_download(cfg_dict)
