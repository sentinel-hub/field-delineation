#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import gc
import logging
import os
import sys
from glob import glob

import fiona
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from fd.utils import multiprocess
from fd.vectorisation import merge_intersecting, split_intersecting

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)


def get_bounds(gdf):
    with fiona.open(gdf) as src:
        return Polygon.from_bounds(*src.bounds)


def _join_overlapping_gdfs(gdf1, gdf2):
    assert gdf1.crs == gdf2.crs, f'The inputs [{gdf1}, {gdf2}] are not in the same CRS!'

    bounds1 = Polygon.from_bounds(*list(gdf1.total_bounds))
    bounds2 = Polygon.from_bounds(*list(gdf2.total_bounds))

    overlap = bounds1.intersection(bounds2)

    non_overlaps1, overlaps1 = split_intersecting(gdf1, overlap)
    non_overlaps2, overlaps2 = split_intersecting(gdf2, overlap)

    intersecting = merge_intersecting(overlaps1, overlaps2)

    out = gpd.GeoDataFrame(pd.concat([non_overlaps1, non_overlaps2, intersecting]), crs=gdf1.crs)
    return out


def join_overlapping(gdf1, gdf2):
    if isinstance(gdf1, str):
        gdf1 = gpd.read_file(gdf1)
        gdf2 = gpd.read_file(gdf2)

    return _join_overlapping_gdfs(gdf1, gdf2)


def joiner(a):
    level, i, df1, df2 = a
    file = f'joined_level_{level}_{i}.gpkg'
    if not os.path.exists(file):
        try:
            res = join_overlapping(df1, df2)
            res.reset_index(drop=True).to_file(file, driver='GPKG')
        except Exception as e:
            raise Exception(e, f'Failed joining {df1} and {df2}!')
    return file


def tester(a):
    level, i, df1, df2 = a
    file = f'joined_level_{level}_{i}.gpkg'
    if not os.path.exists(file):
        raise Exception(f'Joined file {file} is missing!')
    return file


def main(files_to_merge, start_level=0):
    level = 0
    while len(files_to_merge) > 2:
        print(f'processing level {level}, {len(files_to_merge)} dfs to join')
        pairs = [(level, i, gdf1, gdf2) for i, (gdf1, gdf2) in
                 enumerate(zip(files_to_merge[0::2], files_to_merge[1::2]))]
        if level < start_level:
            results = multiprocess(tester, pairs, total=len(pairs), max_workers=24)
        else:
            results = multiprocess(joiner, pairs, total=len(pairs), max_workers=24)
        if len(pairs) % 2:
            results.append(files_to_merge[-1])
        files_to_merge = results
        level = level + 1
        gc.collect()
    
    final = joiner((level, 0, *files_to_merge))
    print(f'result in {final}')


if __name__ == "__main__":
    rows = sorted(glob('./contours/FULL/utm32633/*row*gpkg'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    main(rows, start_level=1)

