#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

from typing import Iterable, Dict, List
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from fd.utils import BaseConfig, prepare_filesystem


@dataclass
class ComputeNormalizationConfig(BaseConfig):
    npz_files_folder: str
    metadata_file: str


def stats_per_npz_ts(npz_file: str, config: ComputeNormalizationConfig) -> Dict[str, np.array]:
    filesystem = prepare_filesystem(config)
    data = np.load(filesystem.openbin(os.path.join(config.npz_files_folder, npz_file), 'rb'), allow_pickle=True)
    features = data['X']

    return {'mean': np.mean(features, axis=(1, 2)),
            'median': np.median(features, axis=(1, 2)),
            'perc_1': np.percentile(features, q=1, axis=(1, 2)),
            'perc_5': np.percentile(features, q=5, axis=(1, 2)),
            'perc_95': np.percentile(features, q=95, axis=(1, 2)),
            'perc_99': np.percentile(features, q=99, axis=(1, 2)),
            'std': np.std(features, axis=(1, 2)),
            'minimum': np.min(features, axis=(1, 2)),
            'maximum': np.max(features, axis=(1, 2)),
            'timestamp': data['timestamps'],
            'patchlet': data['eopatches']
            }


def concat_npz_results(stat: str, results: List[Dict[str, np.array]]) -> np.array:
    return np.concatenate([x[stat] for x in results])


def create_per_band_norm_dataframe(concatenated_stats: Dict[str, np.array], stats_keys: Iterable[str],
                                   identifier_keys: Iterable[str]) -> pd.DataFrame:
    norm_df_dict = {}
    n_bands = concatenated_stats[stats_keys[0]].shape[-1]
    for stat in stats_keys:
        for band in range(0, n_bands):
            norm_df_dict[f'{stat}_b{band}'] = concatenated_stats[stat][..., band]
    for identifier in identifier_keys:
        norm_df_dict[f'{identifier}'] = concatenated_stats[identifier]

    return pd.DataFrame(norm_df_dict)
