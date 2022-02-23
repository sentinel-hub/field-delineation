#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import datetime
from dataclasses import dataclass
from typing import Tuple, List, Union

import numpy as np
from PIL import Image
from fs.errors import ResourceNotFound
from skimage.filters import rank
from skimage.morphology import disk, dilation
from tqdm.auto import tqdm

from eolearn.core import LoadTask, SaveTask, EOTask, EOPatch, FeatureType, OverwritePermission, LinearWorkflow
from eolearn.io import ExportToTiffTask
from sentinelhub import parse_time
from .utils import BaseConfig, set_sh_config


@dataclass
class PostProcessConfig(BaseConfig):
    eopatches_folder: str
    tiffs_folder: str
    time_intervals: dict
    feature_extent: Tuple[FeatureType, str]
    feature_boundary: Tuple[FeatureType, str]
    model_version: str
    max_cloud_coverage: float = 0.05
    percentile: int = 50
    scale_factor: int = 2
    disk_size: int = 2
    masked_temporal_merging: bool = False
    disk_size_masking: int = 5


def upscale_and_rescale(array: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    """ Upscale a given array by a given scale factor using bicubic interpolation """
    assert array.ndim == 2

    height, width = array.shape

    rescaled = np.array(Image.fromarray(array).resize((width * scale_factor, height * scale_factor), Image.BICUBIC))
    rescaled = (rescaled - np.min(rescaled)) / (np.max(rescaled) - np.min(rescaled))

    assert np.sum(~np.isfinite(rescaled)) == 0

    return rescaled


def smooth(array: np.ndarray, disk_size: int = 2) -> np.ndarray:
    """ Blur input array using a disk element of a given disk size """
    assert array.ndim == 2

    smoothed = rank.mean(array, selem=disk(disk_size).astype(np.float32)).astype(np.float32)
    smoothed = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))

    assert np.sum(~np.isfinite(smoothed)) == 0

    return smoothed


class CombineUpsample(EOTask):
    def __init__(self, feature_extent: Tuple[FeatureType, str], feature_boundary: Tuple[FeatureType, str],
                 feature_output: Tuple[FeatureType, str], scale_factor: int = 2, disk_size: int = 2):
        """ Combine extent and boundary pseudo-probabilities and upscale

        :param feature_extent: Feature in eopatch holding extent predictions
        :param feature_boundary: Feature in eopatch holding boundary predictions
        :param feature_output: Feature in eopatch where output will be written
        :param scale_factor: Single scale factor for upsampling. The final factor is twice this value, since two
                            upscaling operations are applied
        :param disk_size: Size of disk used in smoothing of the pseudo-probabilities
        """
        self.feature_extent = next(self._parse_features(feature_extent)())
        self.feature_boundary = next(self._parse_features(feature_boundary)())
        self.feature_output = next(self._parse_features(feature_output)())
        self.scale_factor = scale_factor
        self.disk_size = disk_size

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """ Run merging and upscaling """
        extent = eopatch[self.feature_extent][..., 0]
        boundary = eopatch[self.feature_boundary][..., 0]

        combined = np.clip(1 + extent - boundary, 0, 2)

        combined = upscale_and_rescale(combined, scale_factor=self.scale_factor)
        combined = smooth(combined, disk_size=self.disk_size)
        combined = upscale_and_rescale(combined, scale_factor=self.scale_factor)
        array = smooth(combined, disk_size=self.disk_size * self.scale_factor)

        eopatch[self.feature_output] = np.expand_dims(array, axis=-1)

        return eopatch


class TemporalMerging(EOTask):
    def __init__(self, feature: Tuple[FeatureType, str], feature_merged: Tuple[FeatureType, str],
                 start: Union[datetime.date, str], end: Union[datetime.date, str], percentile: int = 50,
                 max_cloud_coverage: float = 0.05, valid_data_fraction: float = 1.0):
        """ Temporally merge predictions within a given time interval using the given percentile

        :param feature: Feature in eopatch holding temporal predictions
        :param feature_merged: Output feature in eopatch with temporally merged predictions
        """
        self.feature = next(self._parse_features(feature)())
        self.feature_merged = next(self._parse_features(feature_merged)())
        self.percentile = percentile
        self.start = parse_time(start)
        self.end = parse_time(end)
        self.max_cloud_coverage = max_cloud_coverage
        self.valid_data_fraction = valid_data_fraction

    def execute(self, eopatch: EOPatch) -> EOPatch:
        features = []

        for i, ts in enumerate(eopatch.timestamp):
            cc = np.mean(eopatch.mask['CLM'][i, ...])
            valid = np.mean(eopatch.mask['IS_DATA'][i, ...])
            # TODO: make this generalise to any valid data fraction
            if self.start <= ts.date() < self.end and \
                    cc < self.max_cloud_coverage and valid == self.valid_data_fraction:
                features.append(eopatch[self.feature][i, ...])

        features = np.stack(features, axis=0)

        eopatch[self.feature_merged] = np.nan_to_num(
            np.nanpercentile(
                np.ma.filled(features, np.nan),
                q=self.percentile, axis=0
            )
        )

        return eopatch


class MaskedTemporalMerging(EOTask):
    def __init__(self, feature: Tuple[FeatureType, str], feature_merged: Tuple[FeatureType, str],
                 start: Union[datetime.date, str], end: Union[datetime.date, str],
                 percentile: int = 50, dilate_size: int = 5):
        """ Temporally merge predictions within a given time interval using the given percentile,
            masking the prediction where there is no data or is masked with clouds

        :param feature: Feature in eopatch holding temporal predictions
        :param feature_merged: Output feature in eopatch with temporally merged predictions
        """
        self.feature = next(self._parse_features(feature)())
        self.feature_merged = next(self._parse_features(feature_merged)())
        self.percentile = percentile
        self.start = parse_time(start)
        self.end = parse_time(end)
        self.disk = disk(dilate_size)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        features = []

        for i, ts in enumerate(eopatch.timestamp):
            if self.start <= ts.date() < self.end:
                masked_data = eopatch.mask['CLM'][i, ...] | ~eopatch.mask['IS_DATA'][i, ...]
                masked_dilated = dilation(masked_data.squeeze(), self.disk)[:, :, np.newaxis] == 255
                features.append(np.ma.array(eopatch[self.feature][i, ...], mask=masked_dilated))

        features = np.ma.stack(features, axis=0)

        eopatch[self.feature_merged] = np.nan_to_num(
            np.nanpercentile(
                np.ma.filled(features, np.nan),
                q=self.percentile, axis=0
            )
        )

        return eopatch


def get_post_processing_workflow(config: PostProcessConfig) -> LinearWorkflow:
    sh_config = set_sh_config(config)

    load_task = LoadTask(path=f's3://{config.bucket_name}/{config.eopatches_folder}',
                         features=[config.feature_extent,
                                   config.feature_boundary,
                                   (FeatureType.MASK, 'CLM'),
                                   (FeatureType.MASK, 'IS_DATA'),
                                   FeatureType.TIMESTAMP,
                                   FeatureType.META_INFO,
                                   FeatureType.BBOX],
                         config=sh_config), 'Load EOPatch'

    if config.masked_temporal_merging:
        merge_extent_tasks = [(MaskedTemporalMerging(
            feature=config.feature_extent,
            feature_merged=(FeatureType.DATA_TIMELESS, f'{config.feature_extent[1]}_{month}'),
            start=_start, end=_end, percentile=config.percentile, dilate_size=config.disk_size_masking),
                               f'Merge EXTENT for {month}')
            for month, (_start, _end) in config.time_intervals.items()
        ]

        merge_boundary_tasks = [(MaskedTemporalMerging(
            feature=config.feature_boundary,
            feature_merged=(FeatureType.DATA_TIMELESS, f'{config.feature_boundary[1]}_{month}'),
            start=_start, end=_end, percentile=config.percentile, dilate_size=config.disk_size_masking),
                                 f'Merge BOUNDARY for {month}')
            for month, (_start, _end) in config.time_intervals.items()
        ]
    else:
        merge_extent_tasks = [(TemporalMerging(feature=config.feature_extent,
                                               feature_merged=(FeatureType.DATA_TIMELESS,
                                                               f'{config.feature_extent[1]}_{month}'),
                                               start=_start, end=_end,
                                               percentile=config.percentile,
                                               max_cloud_coverage=config.max_cloud_coverage),
                               f'Merge EXTENT for {month}')
                              for month, (_start, _end) in config.time_intervals.items()]

        merge_boundary_tasks = [(TemporalMerging(feature=config.feature_boundary,
                                                 feature_merged=(FeatureType.DATA_TIMELESS,
                                                                 f'{config.feature_boundary[1]}_{month}'),
                                                 start=woy_start, end=woy_end,
                                                 percentile=config.percentile,
                                                 max_cloud_coverage=config.max_cloud_coverage),
                                 f'Merge BOUNDARY for {month}')
                                for month, (woy_start, woy_end) in config.time_intervals.items()]

    combine_tasks = [(CombineUpsample(
        feature_extent=(FeatureType.DATA_TIMELESS, f'{config.feature_extent[1]}_{month}'),
        feature_boundary=(FeatureType.DATA_TIMELESS, f'{config.feature_boundary[1]}_{month}'),
        feature_output=(FeatureType.DATA_TIMELESS, f'PREDICTED_{config.model_version}_{month}'),
        scale_factor=config.scale_factor, disk_size=config.disk_size), f'Combine masks for {month}')
        for month in config.time_intervals]

    save_task = SaveTask(path=f's3://{config.bucket_name}/{config.eopatches_folder}',
                         features=[(FeatureType.DATA_TIMELESS, f'{config.feature_extent[1]}_{month}')
                                   for month in config.time_intervals] +
                                  [(FeatureType.DATA_TIMELESS, f'{config.feature_boundary[1]}_{month}')
                                   for month in config.time_intervals] +
                                  [(FeatureType.DATA_TIMELESS, f'PREDICTED_{config.model_version}_{month}')
                                   for month in config.time_intervals],
                         overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
                         config=sh_config), 'Save Task'

    export_tasks = [(ExportToTiffTask(feature=(FeatureType.DATA_TIMELESS, f'PREDICTED_{config.model_version}_{month}'),
                                      folder=f's3://{config.bucket_name}/{config.tiffs_folder}/{month}/',
                                      compress='DEFLATE', image_dtype=np.float32), f'Export tiffs for {month}')
                    for month in config.time_intervals]

    workflow = LinearWorkflow(load_task, *merge_extent_tasks, *merge_boundary_tasks,
                              *combine_tasks, save_task, *export_tasks)

    return workflow


def get_exec_args(workflow: LinearWorkflow, eopatch_list: List[str], config: PostProcessConfig) -> List[dict]:
    """ Utility function to get execution arguments """
    exec_args = []
    tasks = workflow.get_tasks()

    load_bbox = LoadTask(path=f's3://{config.bucket_name}/{config.eopatches_folder}', features=[FeatureType.BBOX])

    for name in tqdm(eopatch_list):
        single_exec_dict = {}

        try:
            eop = load_bbox.execute(eopatch_folder=name)

            for task_name, task in tasks.items():
                if isinstance(task, ExportToTiffTask):
                    single_exec_dict[task] = dict(filename=f'{name}-{eop.bbox.crs.epsg}.tiff')

                if isinstance(task, (LoadTask, SaveTask)):
                    single_exec_dict[task] = dict(eopatch_folder=name)

            exec_args.append(single_exec_dict)

        except ResourceNotFound as exc:
            print(f'{name} - {exc}')

    return exec_args
