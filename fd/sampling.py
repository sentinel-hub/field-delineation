#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import geopandas as gpd
import numpy as np

from eolearn.core import FeatureType, EOPatch, EOTask, FeatureTypeSet
from fd.utils import BaseConfig, prepare_filesystem
from sentinelhub import BBox

LOGGER = logging.getLogger(__name__)


@dataclass
class SamplingConfig(BaseConfig):
    eopatches_location: str
    output_path: str
    sample_positive: bool = True
    grid_definition_file: str = None
    area_geometry_file: str = None
    mask_feature_name: str = 'EXTENT'
    buffer: int = 0
    patch_size: int = 256
    num_samples: int = 10
    max_retries: int = 10
    fraction_valid: float = 0.4
    sampled_feature_name: str = 'BANDS'
    cloud_coverage: float = 0.05


class SamplePatchlets(EOTask):
    """
    The task samples patchlets of a certain size in a given timeless feature different from no valid data value with
    a certain percentage of valid pixels.
    """

    INVALID_DATA_FRACTION = 0.0
    S2_RESOLUTION = 10

    def __init__(self, feature: Tuple[FeatureType, str], buffer: int, patch_size: int, num_samples: int,
                 max_retries: int, sample_features: List[ Tuple[FeatureType, str]], fraction_valid: float = 0.2,
                 no_data_value: int = 0, sample_positive: bool = True, cloud_coverage: float = 0.05):
        """ Task to sample pixels from a reference timeless raster mask, excluding a no valid data value
        """
        self.feature_type, self.feature_name, self.new_feature_name = next(
            self._parse_features(feature, new_names=True,
                                 default_feature_type=FeatureType.MASK_TIMELESS,
                                 allowed_feature_types={FeatureType.MASK_TIMELESS},
                                 rename_function='{}_SAMPLED'.format)())
        self.max_retries = max_retries
        self.fraction = fraction_valid
        self.no_data_value = no_data_value
        self.sample_features = self._parse_features(sample_features)
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.buffer = buffer
        self.sample_positive = sample_positive
        self.cloud_coverage = cloud_coverage

    def _get_clear_indices(self, clm: np.array, vld: np.array) -> List[int]:
        idxs = []
        for i, (clm_mask, vld_mask) in enumerate(zip(clm, vld)):
            num_cloudy_px = clm_mask.sum()
            num_all_px = np.prod(clm_mask.shape[1:3])
            num_invalid_px = np.sum(~vld_mask.astype(bool))

            if (num_cloudy_px / num_all_px < self.cloud_coverage) and (num_invalid_px == self.INVALID_DATA_FRACTION):
                idxs.append(i)

        return idxs

    def _area_fraction_condition(self, ratio: float) -> bool:
        return ratio < self.fraction if self.sample_positive else ratio > self.fraction

    def execute(self, eopatch: EOPatch, seed: int = None) -> List[EOPatch]:
        timestamps = np.array(eopatch.timestamp)
        mask = eopatch[self.feature_type][self.feature_name].squeeze()
        n_rows, n_cols = mask.shape

        if mask.ndim != 2:
            raise ValueError('Invalid shape of sampling reference map.')

        np.random.seed(seed)
        eops_out = []

        for patchlet_num in range(0, self.num_samples):
            ratio = 0.0 if self.sample_positive else 1
            retry_count = 0
            new_eopatch = EOPatch(timestamp=eopatch.timestamp)
            while self._area_fraction_condition(ratio) and retry_count < self.max_retries:
                row = np.random.randint(self.buffer, n_rows - self.patch_size - self.buffer)
                col = np.random.randint(self.buffer, n_cols - self.patch_size - self.buffer)
                patchlet = mask[row:row + self.patch_size, col:col + self.patch_size]
                ratio = np.sum(patchlet != self.no_data_value) / self.patch_size ** 2
                retry_count += 1

            if retry_count == self.max_retries:
                LOGGER.warning(f'Could not determine an area with good enough ratio of valid sampled pixels for '
                               f'patchlet number: {patchlet_num}')
                continue

            clm_patchlet = eopatch.mask['CLM'][:, row:row + self.patch_size, col:col + self.patch_size, :]
            valid_patchlet = eopatch.mask['IS_DATA'][:, row:row + self.patch_size,
                             col:col + self.patch_size, :]
            idxs = self._get_clear_indices(clm_patchlet, valid_patchlet)
            new_eopatch.timestamp = list(timestamps[idxs])

            for feature_type, feature_name in self.sample_features(eopatch):
                if feature_type in FeatureTypeSet.RASTER_TYPES.intersection(FeatureTypeSet.SPATIAL_TYPES):
                    feature_data = eopatch[feature_type][feature_name]
                    if feature_type.is_time_dependent():
                        sampled_data = feature_data[:, row:row + self.patch_size, col:col + self.patch_size, :]
                        sampled_data = sampled_data[idxs]
                    else:
                        sampled_data = feature_data[row:row + self.patch_size, col:col + self.patch_size, :]

                    new_eopatch[feature_type][f'{feature_name}'] = sampled_data

            patchlet_loc = np.array([row, col, self.patch_size])
            new_eopatch[FeatureType.SCALAR_TIMELESS][f'PATCHLET_LOC'] = patchlet_loc
            r, c, s = patchlet_loc
            new_eopatch.bbox = BBox(((eopatch.bbox.min_x + self.S2_RESOLUTION * c,
                                      eopatch.bbox.max_y - self.S2_RESOLUTION * (r + s)),
                                     (eopatch.bbox.min_x + self.S2_RESOLUTION * (c + s),
                                      eopatch.bbox.max_y - self.S2_RESOLUTION * r)),
                                    eopatch.bbox.crs)

            eops_out.append(new_eopatch)
        return eops_out


def prepare_eopatches_paths(sampling_config: SamplingConfig) -> List[str]:
    eopatches_paths = [os.path.join(sampling_config.eopatches_location, eop_name)
                       for eop_name in prepare_filesystem(sampling_config).listdir(sampling_config.eopatches_location)]

    if not sampling_config.sample_positive:
        area_geometry = gpd.read_file(sampling_config.area_geometry_file)
        eopatches_geometry = gpd.read_file(sampling_config.grid_definition_file)[['name', 'geometry']]
        eop_fully_in_area = [eop for eop, geom in eopatches_geometry.values if area_geometry.contains(geom).values[0]]
        eopatches_paths = [x for x in eopatches_paths if os.path.basename(x) in eop_fully_in_area]

    return eopatches_paths


def sample_patch(eop_path: str, sampling_config: SamplingConfig) -> None:
    filesystem = prepare_filesystem(sampling_config)
    task = SamplePatchlets(feature=(FeatureType.MASK_TIMELESS, sampling_config.mask_feature_name),
                           buffer=sampling_config.buffer,
                           patch_size=sampling_config.patch_size,
                           num_samples=sampling_config.num_samples,
                           max_retries=sampling_config.max_retries,
                           fraction_valid=sampling_config.fraction_valid,
                           sample_features=[(FeatureType.DATA, sampling_config.sampled_feature_name),
                                            (FeatureType.MASK_TIMELESS, 'EXTENT'),
                                            (FeatureType.MASK_TIMELESS, 'BOUNDARY'),
                                            (FeatureType.DATA_TIMELESS, 'DISTANCE')],
                           sample_positive=sampling_config.sample_positive)

    eop_name = os.path.basename(eop_path)
    LOGGER.info(f'Processing eop: {eop_name}')
    try:
        eop = EOPatch.load(eop_path, filesystem=filesystem, lazy_loading=True)
        patchlets = task.execute(eop)
        for i, patchlet in enumerate(patchlets):
            patchlet.save(os.path.join(sampling_config.output_path, f'{eop_name}_{i}'), filesystem=filesystem)
    except KeyError as e:
        LOGGER.error(f'Key error. Could not find key: {e}')
    except ValueError as e:
        LOGGER.error(f'Value error. Value does not exist: {e}')
