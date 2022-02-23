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
import argparse
import logging

import geopandas as gpd

from fd.vectorisation import MergeUTMsConfig, utm_zone_merging
from fd.utils import LogFileFilter, mgrs_to_utm


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def merge_zones(config: dict): 
    """ Utility code to merge results from 2 different UTM zones

    Args:
        config (dict): Dictionary with paramters required for UTM zone merging
    """
    merging_config = MergeUTMsConfig(
            bucket_name=config['bucket_name'], 
            aws_access_key_id=config['aws_access_key_id'],
            aws_secret_access_key=config['aws_secret_access_key'],
            aws_region=config['aws_region'],
            time_intervals=config['time_intervals'],
            utms=config['utms'],
            contours_dir=config['contours_dir'], 
            resulting_crs=config['resulting_crs'], 
            max_area=config['max_area'], 
            simplify_tolerance=config['simplify_tolerance'], 
            n_workers=config['n_workers']
        )

    LOGGER.info('Read grid definition file')
    grid_definition = gpd.read_file(config['grid_definition_file'])
    grid_definition['crs'] = grid_definition['name'].apply(lambda name: f'{mgrs_to_utm(name).epsg}')
    
    utm_geoms = [grid_definition[grid_definition['crs']==crs_name].geometry.unary_union 
                for crs_name in merging_config.utms]
    
    LOGGER.info('Find overlapping tiles')
    # TODO: extend to multiple UTM zones
    overlap = utm_geoms[0].intersection(utm_geoms[1]).buffer(config['overlap_buffer'])
    tiled_overlap = grid_definition[grid_definition.intersects(overlap)].unary_union.buffer(config['overlap_buffer'])

    zones = gpd.GeoDataFrame(geometry=[g for g in grid_definition[~grid_definition.intersects(tiled_overlap)].buffer(config['zone_buffer']).unary_union],
                         crs=grid_definition.crs)
    zones['crs'] = zones.geometry.apply(lambda g: grid_definition[grid_definition.intersects(g)]['crs'].unique()[0])

    overlap_df = gpd.GeoDataFrame(geometry=[tiled_overlap], crs=grid_definition.crs)

    LOGGER.info('Merge results for different UTM zones')
    utm_zone_merging(merging_config, overlap_df, zones)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge results for 2 UTM zones into one.\n")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file with parameters required for UTM merging",
        required=True
    )
    args = parser.parse_args()
    LOGGER.info(f'Reading configuration from {args.config}')
    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    merge_zones(cfg_dict)
