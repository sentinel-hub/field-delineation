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

import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm

from eolearn.core import FeatureType

from fd.utils import prepare_filesystem, LogFileFilter
from fd.prediction import PredictionConfig, run_prediction_on_eopatch
from fd.prediction import load_model, load_metadata


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def process_eopatches(fn, eopatches, **kwargs): 
    results = [] 
    for eopatches_path in tqdm(eopatches): 
        results.append(fn(eopatches_path, **kwargs))
    return results


def run_prediction(config: dict):
    """Utility function to run predictions

    Args:
        config (dict): Dictionary with parameters required for running predictions
    """

    reference_extent = (FeatureType(config['reference_extent'][0]), config['reference_extent'][1]) if 'reference_extent' in config else None
    reference_boundary = (FeatureType(config['reference_boundary'][0]), config['reference_boundary'][1]) if 'reference_boundary' in config else None
    reference_distance = (FeatureType(config['reference_distance'][0]), config['reference_distance'][1]) if 'reference_distance' in config else None

    prediction_config = PredictionConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config['aws_access_key_id'], 
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'],
        eopatches_folder=config['eopatches_folder'],
        feature_bands=(FeatureType(config['feature_bands'][0]), config['feature_bands'][1]),
        feature_extent=(FeatureType(config['feature_extent'][0]), config['feature_extent'][1]),
        feature_boundary=(FeatureType(config['feature_boundary'][0]), config['feature_boundary'][1]),
        feature_distance=(FeatureType(config['feature_distance'][0]), config['feature_distance'][1]),
        reference_extent=reference_extent,
        reference_boundary=reference_boundary,
        reference_distance=reference_distance,
        model_path=config['model_path'],
        model_name=config['model_name'],
        model_version=config['model_version'],
        temp_model_path=config['temp_model_path'],
        normalise=config['normalise'],
        height=config['height'],
        width=config['width'],
        pad_buffer=config['pad_buffer'],
        crop_buffer=config['crop_buffer'],
        n_channels=config['n_channels'],
        n_classes=config['n_classes'],
        metadata_path=config['metadata_path'],
        batch_size=config['batch_size'])

    filesystem = prepare_filesystem(prediction_config) 

    LOGGER.info('Load normalisation factors')
    normalisation_factors = load_metadata(filesystem, prediction_config)

    LOGGER.info('Load grid definition')
    grid_definition = gpd.read_file(config['grid_filename'])

    eopatches_list = grid_definition.name.values

    LOGGER.info('Load model')
    model = load_model(filesystem=filesystem, config=prediction_config)

    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    LOGGER.info('Running predictions')
    status = process_eopatches(run_prediction_on_eopatch, 
                               eopatches_list, 
                               config=prediction_config,
                               filesystem=filesystem,
                               model=model,
                               normalisation_factors=normalisation_factors)

    LOGGER.info('Check status of prediction')
    status_df = pd.DataFrame(status)
    LOGGER.info(f'{len(status_df)} total eopatches, {len(status_df[status_df.status!="Success"])} failed')
    LOGGER.info(f'Failed EOPatches are {status_df[status_df.status!="Success"].name}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run predictions on EOPatches.\n")

    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file with parameters required for predictions", 
        required=True
    )
    args = parser.parse_args()

    LOGGER.info(f'Reading configuration from {args.config}')

    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    run_prediction(cfg_dict)
