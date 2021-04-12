#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

"""
Tasks used for batch results handling
"""
import ast
import json
from datetime import datetime

from dataclasses import dataclass
from typing import List, Tuple, Union

import fs
import numpy as np

from eolearn.core import EOPatch, EOTask, FeatureType, EOWorkflow, SaveTask, MergeFeatureTask, RemoveFeature, \
    RenameFeature, OverwritePermission, LinearWorkflow
from eolearn.core.fs_utils import get_base_filesystem_and_path
from eolearn.io import ImportFromTiff

from s2cloudless import S2PixelCloudDetector

from .utils import BaseConfig, set_sh_config


@dataclass
class TiffsToEopatchConfig(BaseConfig):
    tiffs_folder: str
    eopatches_folder: str
    band_names: List[str]
    mask_name: str
    data_name: str = 'BANDS'
    is_data_mask: str = 'IS_DATA'
    clp_name: str = 'CLP'
    clm_name: str = 'CLM'


class AddTimestampsUpdateTime(EOTask):

    """ Task to set eopatch timestamps """

    def __init__(self, path: str):
        """
        :param path: Path to folder containing the tiles
        """
        self.path = path

    def _get_valid_dates(self, tile_name: str, filename: str = 'userdata.json') -> List[datetime]:
        """
        :param tile_name: Name of the tile to process
        :param filename: Name of the json file containing timestamps
        """
        filesystem, relative_path = get_base_filesystem_and_path(self.path)
        full_path = fs.path.join(relative_path, tile_name, filename)

        decoded_data = filesystem.readtext(full_path, encoding='utf-8')
        parsed_data = json.loads(decoded_data)
        dates = ast.literal_eval(parsed_data['dates'])

        return dates

    def execute(self, eopatch: EOPatch, *, tile_name: str) -> EOPatch:
        """
        Execute method to remove invalid features based on available timestamps, and add
        those timestamps to the eopatch. The process is applied to the data and to the
        datamask.

        :param eopatch: Name of the eopatch to process
        :param tile_name: Name of the tile to process
        """
        dates = self._get_valid_dates(tile_name)
        eopatch.timestamp = dates
        return eopatch


class RearrangeBands(EOTask):
    """ Rearrange feature's axis in order to change first and last axis,
    this is done so that the first axis is the temporal axis"""

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        Execute method to move exchange last and first axis for each feature
        and for the datamask

        :param eopatch: Name of the eopatch to process
        """
        for band in eopatch.data:
            eopatch.data[band] = np.swapaxes(eopatch.data[band], 0, -1)

        for mask in eopatch.mask:
            eopatch.mask[mask] = np.swapaxes(eopatch.mask[mask], 0, -1)

        return eopatch


class DeleteFiles(EOTask):
    """ Delete files"""

    def __init__(self, path: str, filenames: List[str]):
        """
        :param path: Path to folder containing the tifs to be deleted
        :param filenames: A list of filenames to delete
        """
        self.path = path
        self.filenames = filenames

    def execute(self, eopatch: Union[EOPatch, str], *, tile_name: str):
        """
        Execute method to delete tiffs relative to the specified tile

        :param eopatch: Name of the eopatch to process
        :param tile_name: Name of the tyle to process
        """
        filesystem, relative_path = get_base_filesystem_and_path(self.path)

        for filename in self.filenames:
            full_path = fs.path.join(relative_path, tile_name, filename)
            filesystem.remove(full_path)


class CloudMasking(EOTask):
    """ Compute cloud mask from SH cloud probability CLP data map """
    def __init__(self, clp_feature: Tuple = (FeatureType.DATA, 'CLP'), clm_feature: Tuple = (FeatureType.MASK, 'CLM'),
                 average_over: int = 24, max_clp: float = 255.):
        """
        :param clp_feature: Feature type and name of input CLP mask
        :param clm_feature: Feature type and name of output CLM mask
        :param average_over: Parameter used ot smooth the CLP data map
        :param max_clp: Maximum value of CLP map used for normalization
        """
        self.clm_feature = next(self._parse_features(clm_feature)())
        self.clp_feature = next(self._parse_features(clp_feature)())
        self.s2_cd = S2PixelCloudDetector(average_over=average_over)
        self.max_clp = max_clp

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """ Compute and add CLM from CLP """
        clc = self.s2_cd.get_mask_from_prob(eopatch[self.clp_feature].squeeze() / self.max_clp)
        eopatch[self.clm_feature] = clc[..., np.newaxis]
        return eopatch


def get_tiffs_to_eopatches_workflow(config: TiffsToEopatchConfig, delete_tiffs: bool = False) -> EOWorkflow:
    """ Set up workflow to ingest tiff files into EOPatches """

    # Set up credentials in sh config
    sh_config = set_sh_config(config)

    import_bands = [(ImportFromTiff((FeatureType.DATA, band),
                                    folder=f's3://{config.bucket_name}/{config.tiffs_folder}',
                                    config=sh_config), f'Import band {band}')
                    for band in config.band_names]
    import_clp = (ImportFromTiff((FeatureType.DATA, config.clp_name),
                                 folder=f's3://{config.bucket_name}/{config.tiffs_folder}',
                                 config=sh_config), f'Import {config.clp_name}')

    import_mask = (ImportFromTiff((FeatureType.MASK, config.mask_name),
                                  folder=f's3://{config.bucket_name}/{config.tiffs_folder}',
                                  config=sh_config), f'Import {config.mask_name}')

    rearrange_bands = (RearrangeBands(), 'Swap time and band axis')
    add_timestamps = (AddTimestampsUpdateTime(f's3://{config.bucket_name}/{config.tiffs_folder}'), 'Load timestamps')

    merge_bands = (MergeFeatureTask(
        input_features={FeatureType.DATA: config.band_names},
        output_feature=(FeatureType.DATA, config.data_name)), 'Merge band features')

    remove_bands = (RemoveFeature(features={FeatureType.DATA: config.band_names}), 'Remove bands')

    rename_mask = (RenameFeature((FeatureType.MASK, config.mask_name, config.is_data_mask)), 'Rename is data mask')

    calculate_clm = (CloudMasking(), 'Get CLM mask from CLP')

    save_task = (SaveTask(path=f's3://{config.bucket_name}/{config.eopatches_folder}', config=sh_config,
                          overwrite_permission=OverwritePermission.OVERWRITE_FEATURES),  'Save EOPatch')

    filenames = [f'{band}.tif' for band in config.band_names] + \
                [f'{config.mask_name}.tif', f'{config.clp_name}.tif', 'userdata.json']
    delete_files = (DeleteFiles(path=config.tiffs_folder, filenames=filenames), 'Delete batch files')

    workflow = [*import_bands,
                import_clp,
                import_mask,
                rearrange_bands,
                add_timestamps,
                merge_bands,
                remove_bands,
                rename_mask,
                calculate_clm,
                save_task]

    if delete_tiffs:
        workflow.append(delete_files)

    return LinearWorkflow(*workflow)


def get_exec_args(workflow: EOWorkflow, eopatch_list: List[str]) -> List[dict]:
    """ Utility function to get execution arguments """
    exec_args = []
    tasks = workflow.get_tasks()

    for name in eopatch_list:
        single_exec_dict = {}

        for task_name, task in tasks.items():
            if isinstance(task, ImportFromTiff):
                tiff_name = task_name.split()[-1]
                path = f'{name}/{tiff_name}.tif'
                single_exec_dict[task] = dict(filename=path)

            if isinstance(task, SaveTask):
                single_exec_dict[task] = dict(eopatch_folder=name)

            if isinstance(task, (AddTimestampsUpdateTime, DeleteFiles)):
                single_exec_dict[task] = dict(tile_name=name)

        exec_args.append(single_exec_dict)

    return exec_args
