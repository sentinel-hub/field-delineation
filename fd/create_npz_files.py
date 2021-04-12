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
import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from dataclasses import dataclass

from eolearn.core import EOPatch
from fd.utils import BaseConfig, prepare_filesystem

LOGGER = logging.getLogger(__name__)


@dataclass
class CreateNpzConfig(BaseConfig):
    patchlets_folder: str
    output_folder: str
    chunk_size: int
    output_dataframe: str


def extract_npys(eopatch_path: str, cfg: CreateNpzConfig) -> Tuple:
    """ Return X, y_boundary, y_extent, y_distance, timestamps, eop_names numpy arrays for this patchlet."""
    filesystem = prepare_filesystem(cfg)
    try:
        eop = EOPatch.load(eopatch_path, filesystem=filesystem, lazy_loading=True)
        n_timestamps = len(eop.timestamp)
        X_data = eop.data['BANDS']
        y_boundary = np.repeat(eop.mask_timeless['BOUNDARY'][np.newaxis, ...], n_timestamps, axis=0)
        y_extent = np.repeat(eop.mask_timeless['EXTENT'][np.newaxis, ...], n_timestamps, axis=0)
        y_distance = np.repeat(eop.data_timeless['DISTANCE'][np.newaxis, ...], n_timestamps, axis=0)
        timestamps = eop.timestamp
        eop_names = np.repeat([eopatch_path], n_timestamps, axis=0)
    except Exception as e:
        LOGGER.error(f"Could not create for {eopatch_path}. Exception {e}")
        return None, None, None, None, None, None
    return X_data, y_boundary, y_extent, y_distance, timestamps, eop_names


def concatenate_npys(results: List[Tuple]) -> Dict[str, np.ndarray]:
    """ Concatenate numpys from each eopatch into one big numpy array"""
    # TODO: This whole process is very RAM inefficient as it relies on fitting the whole training dataset into RAM
    # TODO: for really big countries will need to be completely rethought. Currently this problem is solved by getting
    # TODO: a bigger instance ...

    results = [x for x in results if x[0] is not None]
    X, y_boundary, y_extent, y_distance, timestamps, eop_names = zip(*results)

    X = np.concatenate(X)
    y_boundary = np.concatenate(y_boundary)
    y_extent = np.concatenate(y_extent)
    y_distance = np.concatenate(y_distance)
    timestamps = np.concatenate(timestamps)
    eop_names = np.concatenate(eop_names)

    npys_dict = {'X': X, 'y_boundary': y_boundary, 'y_extent': y_extent, 'y_distance': y_distance,
                 'timestamps': timestamps, 'eop_names': eop_names}

    return npys_dict


def save_into_chunks(config: CreateNpzConfig, npys_dict: Dict[str, np.ndarray]) -> None:
    eopatches = [os.path.basename("_".join(x.split("_")[:-1])) for x in npys_dict['eop_names']]
    filesystem = prepare_filesystem(config)
    chunk_size = config.chunk_size
    dfs = []

    if not filesystem.isdir(config.output_folder):
        filesystem.makedirs(config.output_folder)
    
    for idx, i in enumerate(range(0, len(npys_dict['X']), chunk_size)):
        filename = f'patchlets_field_delineation_{idx}'
        np.savez(filesystem.openbin(os.path.join(config.output_folder, f'{filename}.npz'), 'wb'),
                 X=npys_dict['X'][i:i + chunk_size],
                 y_boundary=npys_dict['y_boundary'][i:i + chunk_size],
                 y_extent=npys_dict['y_extent'][i:i + chunk_size],
                 y_distance=npys_dict['y_distance'][i:i + chunk_size],
                 timestamps=npys_dict['timestamps'][i:i + chunk_size],
                 eopatches=npys_dict['eop_names'][i:i + chunk_size])

        dfs.append(pd.DataFrame(dict(chunk=[f'{filename}.npz'] * len(npys_dict['eop_names'][i:i + chunk_size]),
                                     eopatch=eopatches[i:i + chunk_size],
                                     patchlet=npys_dict['eop_names'][i:i + chunk_size],
                                     chunk_pos=list(range(0, len(eopatches[i:i + chunk_size]))),
                                     timestamp=npys_dict['timestamps'][i:i + chunk_size])))

    metadata_dir = os.path.dirname(config.output_dataframe)
    if not filesystem.isdir(metadata_dir):
        filesystem.makedirs(metadata_dir)
        
    pd.concat(dfs).to_csv(filesystem.open(config.output_dataframe, 'w'), index=False)
