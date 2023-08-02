#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

from typing import Tuple

from dataclasses import dataclass

import pyproj
from shapely.ops import transform

import psycopg2

import geopandas as gpd

import numpy as np
from skimage.morphology import binary_dilation, disk
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

from sentinelhub import SHConfig

from eolearn.core import FeatureType, EOPatch, EOTask, EOWorkflow, LinearWorkflow, \
    LoadTask, SaveTask, OverwritePermission
from eolearn.geometry import VectorToRaster
from eolearn.io.geometry_io import VectorImportTask


from .utils import BaseConfig, set_sh_config


@dataclass
class GsaaToEopatchConfig(BaseConfig):
    reference_file_path: str
    vector_feature: Tuple[FeatureType, str]
    extent_feature: Tuple[FeatureType, str]
    boundary_feature: Tuple[FeatureType, str]
    distance_feature: Tuple[FeatureType, str]
    eopatches_folder: str
    buffer_poly: int = -10
    no_data_value: int = 0
    width: int = 1100
    height: int = 1100
    disk_radius: int = 2


class DB2Vector(EOTask):
    """
    Reads vectors to EOPatch from a local postgre db.
    """

    def __init__(self, database: str, user: str, password: str, host: str,
                 port: str, crs: pyproj.crs, vector_output_feature: Tuple):
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.crs = crs
        self.out_vector = next(self._parse_features(vector_output_feature)())

    def execute(self, eopatch: EOPatch) -> EOPatch:
        utm_crs = eopatch.bbox.crs.pyproj_crs()
        project = pyproj.Transformer.from_proj(utm_crs, self.crs)

        query_bbox = transform(project.transform, eopatch.bbox.geometry)

        spatial_query = 'select * from gsaa where ST_Intersects(ST_GeomFromText((%s), (%s)), geom);'
        parameters = (query_bbox.wkt, self.crs.to_epsg())

        with psycopg2.connect(database=self.database, user=self.user, password=self.password,
                              host=self.host, port=self.port) as con:
            df = gpd.GeoDataFrame.from_postgis(spatial_query, con, geom_col='geom', params=parameters)
            df = df.to_crs(utm_crs).rename(columns={'geom': 'geometry'}).set_geometry('geometry')

        eopatch[self.out_vector] = df

        return eopatch


class Extent2Boundary(EOTask):
    """
    Adds boundary mask from extent mask using binary dilation
    """

    def __init__(self, extent_feature: Tuple[FeatureType, str],
                 boundary_feature: Tuple[FeatureType, str], structure: np.ndarray = None):
        self.extent_feature = next(self._parse_features(extent_feature)())
        self.boundary_feature = next(self._parse_features(boundary_feature)())
        self.structure = structure

    def execute(self, eopatch):
        extent_mask = eopatch[self.extent_feature].squeeze(axis=-1)
        boundary_mask = binary_dilation(extent_mask, selem=self.structure) - extent_mask
        eopatch[self.boundary_feature] = boundary_mask[..., np.newaxis]

        return eopatch


class Extent2Distance(EOTask):
    """
    Adds boundary mask from extent mask using binary dilation
    """

    def __init__(self, extent_feature: Tuple[FeatureType, str],
                 distance_feature: Tuple[FeatureType, str], normalize: bool = True):
        self.extent_feature = next(self._parse_features(extent_feature)())
        self.distance_feature = next(self._parse_features(distance_feature)())
        self.normalize = normalize

    def execute(self, eopatch):
        extent_mask = eopatch[self.extent_feature].squeeze(axis=-1)

        distance = distance_transform_edt(extent_mask)

        if not self.normalize:
            eopatch[self.distance_feature] = distance[..., np.newaxis]

            return eopatch

        conn_comp = label(extent_mask, background=0)
        unique_comp = np.unique(conn_comp)
        normalised = np.zeros(distance.shape, dtype=np.float32)

        for uc in unique_comp:
            if uc != 0:
                conn_comp_mask = conn_comp == uc
                normalised[conn_comp_mask] += distance[conn_comp_mask] / np.max(distance[conn_comp_mask])

        eopatch[self.distance_feature] = normalised[..., np.newaxis]

        return eopatch


def get_gsaa_to_eopatch_workflow(config: GsaaToEopatchConfig) -> EOWorkflow:
    # set up AWS credentials
    sh_config = set_sh_config(config)

    # load patch
    load_task = LoadTask(path=f's3://{config.bucket_name}/{config.eopatches_folder}', config=sh_config)
    # add original vectors to patch
    vec2vec = DB2Vector(database=config.database,
                        user=config.user, password=config.password,
                        host=config.host, port=config.port, crs=config.crs,
                        vector_output_feature=config.vector_feature)
    # get extent mask from vector
    vec2ras = VectorToRaster(config.vector_feature,
                             config.extent_feature,
                             values=1, raster_shape=(config.width, config.height),
                             no_data_value=config.no_data_value,
                             buffer=config.buffer_poly, write_to_existing=False)
    # get boundary mask from extent mask
    ras2bound = Extent2Boundary(config.extent_feature,
                                config.boundary_feature,
                                structure=disk(config.disk_radius))
    # get distance from extent mask
    ras2dist = Extent2Distance(config.extent_feature,
                               config.distance_feature,
                               normalize=True)
    # save new features
    save_task = SaveTask(path=f's3://{config.bucket_name}/{config.eopatches_folder}',
                         features=[config.vector_feature,
                                   config.extent_feature,
                                   config.boundary_feature,
                                   config.distance_feature],
                         overwrite_permission=OverwritePermission.OVERWRITE_FEATURES, config=sh_config)

    return LinearWorkflow(load_task, vec2vec, vec2ras, ras2bound, ras2dist, save_task)
