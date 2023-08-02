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
from argparse import RawTextHelpFormatter
import warnings

import geopandas as gpd
from tqdm.auto import tqdm

from sentinelhub import CRS

from eolearn.core import FeatureType, EOExecutor

from fd.gsaa_to_eopatch import GsaaToEopatchConfig, get_gsaa_to_eopatch_workflow
from fd.utils import LogFileFilter


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def rasterise_gsaa(config: dict):
    """ Add GSAA vector data to eopatches and resterise them

    Args:
        config (dict): Configuration file with rasterisation parameters
    """

    gsaa_to_eops_config = GsaaToEopatchConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'],
        reference_file_path=config['reference_file_path'],
        eopatches_folder=config['eopatches_folder'],
        vector_feature=(FeatureType(config['vector_feature'][0]), config['vector_feature'][1]),
        extent_feature=(FeatureType(config['extent_feature'][0]), config['extent_feature'][1]),
        boundary_feature=(FeatureType(config['boundary_feature'][0]), config['boundary_feature'][1]),
        distance_feature=(FeatureType(config['distance_feature'][0]), config['distance_feature'][1]),
        height=config['height'],
        width=config['width'],
        buffer_poly=config['buffer_poly'],
        no_data_value=config['no_data_value'],
        disk_radius=config['disk_radius']
    )

    LOGGER.info('Read grid definition file')
    grid_definition = gpd.read_file(config['grid_filename'])

    eopatches_list = grid_definition.name.values

    workflow = get_gsaa_to_eopatch_workflow(gsaa_to_eops_config)

    tasks = workflow.get_tasks()

    LOGGER.info('Create Execution arguments')
    exec_args = []
    for eopatch_name in tqdm(eopatches_list, total=len(eopatches_list)):
        single_exec_dict = {}
        single_exec_dict[tasks['LoadTask']] = dict(eopatch_folder=f'{eopatch_name}')
        single_exec_dict[tasks['SaveTask']] = dict(eopatch_folder=f'{eopatch_name}')
        exec_args.append(single_exec_dict)

    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    LOGGER.info('Execution rasterization workflow')
    executor = EOExecutor(workflow, exec_args, save_logs=True, logs_folder='.')
    executor.run(workers=config['max_workers'])
    executor.make_report()

    LOGGER.info(f'Report was saved to location: {executor.get_report_filename()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add GSAA reference data to eopatches.\n\n"
        "This script assumes the GSAA vectors are served in a postgis database. Do the following to set-up the database.\n\n"
        "$ docker pull kartoza/postgis\n"
        "$ docker run --name ""postgis"" -p 25431:5432 -e POSTGRES_USER=niva -e POSTGRES_PASS=n1v4 -e POSTGRES_DBNAME=gisdb -e POSTGRES_MULTIPLE_EXTENSIONS=postgis,hstore -d -t kartoza/postgis\n"
        "$ sudo apt install postgis\n"
        "$ sudo apt install gdal-bin\n"
        "$ ogr2ogr -t_srs epsg:4326 -lco ENCODING=UTF-8 -f 'Esri Shapefile' gsaa_4326.shp bucket/Declared_parcels_2019_03_S4C.shp\n"
        "$ shp2pgsql -s 4326 -I gsaa_4326.shp gsaa > gsaa.sql\n"
        "$ psql -h localhost -U niva -p 25431 -d gisdb -f gsaa.sql", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file with rasterisation parameters", 
        required=True
    )
    args = parser.parse_args()

    LOGGER.info(f'Reading configuration from {args.config}')
    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    rasterise_gsaa(cfg_dict)
