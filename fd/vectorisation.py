#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import copy
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from glob import glob
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from fs.copy import copy_dir
from geopandas.tools import sjoin
from lxml import etree
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm.auto import tqdm as tqdm

from sentinelhub import CRS
from .utils import BaseConfig, multiprocess, prepare_filesystem

LOGGER = logging.getLogger(__name__)


@dataclass
class VectorisationConfig(BaseConfig):
    tiffs_folder: str
    time_intervals: List[str]
    utms: List[str]
    shape: Tuple[int, int]
    buffer: Tuple[int, int]
    weights_file: str
    vrt_dir: str
    predictions_dir: str
    contours_dir: str
    max_workers: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 10
    threshold: float = 0.6
    cleanup: bool = True
    skip_existing: bool = True
    rows_merging: bool = True


@dataclass
class MergeUTMsConfig(BaseConfig):
    time_intervals: List[str]
    utms: List[str]
    contours_dir: str
    resulting_crs: str
    max_area: float = None
    simplify_tolerance: float = 2.5
    n_workers: int = 34


def average_function(no_data: Union[int, float] = 0, round_output: bool =False) -> str:
    """ A Python function that will be added to VRT and used to calculate weighted average over overlaps

    :param no_data: no data pixel value (default = 0)
    :param round_output: flag to round the output (to 0 decimals). Useful when the final result will be in Int.
    :return: Function (as a string)
    """
    rounding = 'out = np.round(out, 0)' if round_output else ''
    return f"""
import numpy as np

def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    p, w = np.split(np.array(in_ar), 2, axis=0)
    n_overlaps = np.sum(p!={no_data}, axis=0)
    w_sum = np.sum(w, axis=0, dtype=np.float32) 
    p_sum = np.sum(p, axis=0, dtype=np.float32) 
    weighted = np.sum(p*w, axis=0, dtype=np.float32)
    out = np.where((n_overlaps>1) & (w_sum>0) , weighted/w_sum, p_sum/n_overlaps)
    {rounding}
    out_ar[:] = out
    
"""


def p_simplify(r, tolerance=2.5):
    """ Helper function to parallelise simplification of geometries """
    return r.geometry.simplify(tolerance)


def p_union(r):
    """ Helper function to parallelise union of geometries """
    return r.l_geom.union(r.r_geom)


def get_weights(shape: Tuple[int, int], buffer: Tuple[int, int], low: float = 0, high: float = 1) -> np.ndarray:
    """ Create weights array

    Function to create a numpy array of dimension, that outputs a linear gradient from low to high from the edges
    to the 2*buffer, and 1 elsewhere.
    """
    weight = np.ones(shape)
    weight[..., :2 * buffer[0]] = np.tile(np.linspace(low, high, 2 * buffer[0]), shape[0]).reshape(
        (shape[0], 2 * buffer[0]))
    weight[..., -2 * buffer[0]:] = np.tile(np.linspace(high, low, 2 * buffer[0]), shape[0]).reshape(
        (shape[0], 2 * buffer[0]))
    weight[:2 * buffer[1], ...] = weight[:2 * buffer[1], ...] * np.repeat(np.linspace(low, high, shape[1]),
                                                                          2 * buffer[1]).reshape(
        (2 * buffer[1], shape[1]))
    weight[-2 * buffer[1]:, ...] = weight[-2 * buffer[1]:, ...] * np.repeat(np.linspace(high, low, 2 * buffer[1]),
                                                                            shape[1]).reshape(
        (2 * buffer[1], shape[1]))
    return weight.astype(np.float32)


def write_vrt(files: List[str], weights_file: str, out_vrt: str, function: Optional[str] = None):
    """ Write virtual raster

    Function that will first build a temp.vrt for the input files, and then modify it for purposes of spatial merging
    of overlaps using the provided function
    """

    if not function:
        function = average_function()

    # build a vrt from list of input files
    gdal_str = f'gdalbuildvrt temp.vrt -b 1 {" ".join(files)}'
    os.system(gdal_str)

    # fix the vrt
    root = etree.parse('temp.vrt').getroot()
    vrtrasterband = root.find('VRTRasterBand')
    rasterbandchildren = list(vrtrasterband)
    root.remove(vrtrasterband)

    dict_attr = {'dataType': 'Float32', 'band': '1', 'subClass': 'VRTDerivedRasterBand'}
    raster_band_tag = etree.SubElement(root, 'VRTRasterBand', dict_attr)

    # Add childern tags to derivedRasterBand tag
    pix_func_tag = etree.SubElement(raster_band_tag, 'PixelFunctionType')
    pix_func_tag.text = 'average'

    pix_func_tag2 = etree.SubElement(raster_band_tag, 'PixelFunctionLanguage')
    pix_func_tag2.text = 'Python'

    pix_func_code = etree.SubElement(raster_band_tag, 'PixelFunctionCode')
    pix_func_code.text = etree.CDATA(function)

    new_sources = []
    for child in rasterbandchildren:
        if child.tag == 'NoDataValue':
            pass
        else:
            raster_band_tag.append(child)
        if child.tag == 'ComplexSource':
            new_source = copy.deepcopy(child)
            new_source.find('SourceFilename').text = weights_file
            new_source.find('SourceProperties').attrib['DataType'] = 'Float32'
            for nodata in new_source.xpath('//NODATA'):
                nodata.getparent().remove(nodata)
            new_sources.append(new_source)

    for new_source in new_sources:
        raster_band_tag.append(new_source)

    os.remove('temp.vrt')

    with open(out_vrt, 'w') as out:
        out.writelines(etree.tounicode(root, pretty_print=True))


def run_contour(col: int, row: int, size: int, vrt_file: str, threshold: float = 0.6,
                contours_dir: str = '.', cleanup: bool = True, skip_existing: bool = True) -> Tuple[str, bool, str]:
    """ Will create a (small) tiff file over a srcwin (row, col, size, size) and run gdal_contour on it. """

    file = f'merged_{row}_{col}_{size}_{size}'
    if skip_existing and os.path.exists(file):
        return file, True, 'Loaded existing file ...'
    try:
        gdal_str = f'gdal_translate --config GDAL_VRT_ENABLE_PYTHON YES -srcwin {col} {row} {size} {size} {vrt_file} {file}.tiff'
        os.system(gdal_str)
        gdal_str = f'gdal_contour -of gpkg {file}.tiff {contours_dir}/{file}.gpkg -i {threshold} -amin amin -amax amax -p'
        os.system(gdal_str)
        if cleanup:
            os.remove(f'{file}.tiff')
        return f'{contours_dir}/{file}.gpkg', True, None
    except Exception as exc:
        return f'{contours_dir}/{file}.gpkg', False, exc


def runner(arg: List):
    """Function that wraps run_contour to be used with sg_utils.postprocessing"""
    return run_contour(*arg)


def unpack_contours(df_filename: str, threshold: float = 0.6) -> gpd.GeoDataFrame:
    """ Convert multipolygon contour row above given threshold into multiple Polygon rows. """
    df = gpd.read_file(df_filename)
    if len(df) <= 2:
        if len(df[df.amax > threshold]):
            return gpd.GeoDataFrame(geometry=[geom for geom in df[df.amax > threshold].iloc[0].geometry], crs=df.crs)
        else:
            return gpd.GeoDataFrame(geometry=[], crs=df.crs)
    raise ValueError(
        f"gdal_contour dataframe {df_filename} has {len(df)} contours, "
        f"but should have maximal 2 entries (one below and/or one above threshold)!")


def split_intersecting(df: gpd.GeoDataFrame, overlap: Polygon) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """ Find entries that overlap with a given polygon """
    index = df.sindex
    possible_matches_index = list(index.intersection(overlap.bounds))
    possible_matches = df.iloc[possible_matches_index]
    precise_matches = possible_matches.intersects(overlap).index

    if len(precise_matches):
        return df[~df.index.isin(precise_matches)].copy(), df[df.index.isin(precise_matches)].copy()

    return df, gpd.GeoDataFrame(geometry=[], crs=df.crs)
    

def merge_intersecting(df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Merge two dataframes of geometries into one """
    multi = unary_union(list(df1.geometry) + list(df2.geometry))
    if multi.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=df1.crs)
    if multi.geom_type == 'Polygon':
        return gpd.GeoDataFrame(geometry=[multi], crs=df1.crs)
    return gpd.GeoDataFrame(geometry=[g for g in multi.geoms], crs=df1.crs)


def concat_consecutive(merged: gpd.GeoDataFrame, previous: gpd.GeoDataFrame, current: gpd.GeoDataFrame,
                       current_offset: Tuple[int, int], overlap_size: Tuple[int, int] = (10, 500),
                       direction: Tuple[int, int] = (490, 0), transform=None) -> Tuple[gpd.GeoDataFrame,
                                                                                       gpd.GeoDataFrame]:
    list_dfs = []
    if merged is not None:
        list_dfs = [merged]

    if not (len(previous) or len(current)):
        if merged is not None:
            return merged, gpd.GeoDataFrame(geometry=[], crs=merged.crs)
        else:
            return merged, gpd.GeoDataFrame(geometry=[])

    x, y = current_offset
    a, b = overlap_size
    overlap_poly = Polygon.from_bounds(*(transform * (x, y)), *(transform * (x + a, y + b)))

    if len(previous) == 0:
        return merged, current

    if len(current) == 0:
        merged = gpd.GeoDataFrame(pd.concat([merged, previous]), crs=previous.crs)
        return merged, gpd.GeoDataFrame(geometry=[], crs=merged.crs)

    previous_non, previous_int = split_intersecting(previous, overlap_poly)
    current_non, current_int = split_intersecting(current, overlap_poly)
    intersecting = merge_intersecting(previous_int, current_int)
    if len(intersecting):
        # check if intersecting "touches" the "right edge", if so, add it to current_non
        x = x + direction[0]
        y = y + direction[1]
        overlap_poly_end = Polygon.from_bounds(*(transform * (x, y)), *(transform * (x + a, y + b)))
        intersecting_ok, intersecting_next = split_intersecting(intersecting, overlap_poly_end)
        merged = gpd.GeoDataFrame(pd.concat(list_dfs + [previous_non, intersecting_ok]), crs=previous.crs)
        intersecting_next = gpd.GeoDataFrame(pd.concat([intersecting_next, current_non]), crs=previous.crs)
        return merged, intersecting_next

    return gpd.GeoDataFrame(pd.concat(list_dfs + [previous_non]), crs=previous.crs), current_non


def _process_row(row: int, vrt_file: str, vrt_dim: Tuple, contours_dir: str = '.', size: int = 500, buff: int = 10,
                 threshold: float = 0.6, cleanup: bool = True, transform=None, skip_existing: bool = True) \
        -> Tuple[str, bool, str]:
    merged_file = f'{contours_dir}/merged_row_{row}.gpkg'
    if skip_existing and os.path.exists(merged_file):
        return merged_file, True, 'Loaded existing file ...'
    try:
        col = 0
        merged = None
        prev_name, finished, exc = run_contour(col, row, size, vrt_file, threshold, contours_dir, cleanup, skip_existing)
        if not finished:
            return merged_file, finished, exc
        prev = unpack_contours(prev_name, threshold=threshold)
        if cleanup:
            os.remove(prev_name)
        while col <= (vrt_dim[0] - size):
            col = col + size - buff
            offset = col, row
            cur_name, finished, exc = run_contour(col, row, size, vrt_file, threshold, contours_dir, cleanup, skip_existing)
            if not finished:
                return merged_file, finished, exc
            cur = unpack_contours(cur_name, threshold=threshold)
            merged, prev = concat_consecutive(merged, prev, cur, offset, (buff, size), (size - buff, 0), transform)
            if cleanup:
                os.remove(cur_name)
        merged = gpd.GeoDataFrame(pd.concat([merged, prev]), crs=prev.crs)
        merged.to_file(merged_file, driver='GPKG')
        return merged_file, True, None
    except Exception as exc:
        return merged_file, False, exc


def merge_rows(rows: List[str], vrt_file: str, size: int = 500, buffer: int = 10) -> gpd.GeoDataFrame:
    with rasterio.open(vrt_file) as src:
        meta = src.meta
        vrt_dim = meta['width'], meta['height']
        transform = meta['transform']

    merged = None
    prev_name = rows[0]
    prev = gpd.read_file(prev_name)
    for ridx, cur_name in tqdm(enumerate(rows[1:], start=1), total=len(rows)-1):
        cur = gpd.read_file(cur_name)
        merged, prev = concat_consecutive(merged, prev, cur, (0, ridx * (size - buffer)), (vrt_dim[0], buffer),
                                          (0, size - buffer), transform)
    merged = gpd.GeoDataFrame(pd.concat([merged, prev]), crs=prev.crs)

    return merged


def spatial_merge_contours(vrt_file: str, contours_dir: str = '.', size: int = 500, buffer: int = 10,
                           threshold: float = 0.6, cleanup: bool = True, skip_existing: bool = True,
                           rows_merging: bool = True, max_workers: int = 4) -> gpd.GeoDataFrame:
    results = process_rows(vrt_file=vrt_file, contours_dir=contours_dir, size=size, buffer=buffer, threshold=threshold,
                           cleanup=cleanup, skip_existing=skip_existing, max_workers=max_workers)

    failed = [(file, excp) for file, finished, excp in results if not finished]
    if len(failed):
        LOGGER.warning('Some rows failed:')
        LOGGER.warning('\n'.join([f'{file}: {excp}' for file, excp in failed]))
        return None

    if rows_merging:
        rows = [file for file, _, _ in results]
        merged = merge_rows(rows, vrt_file=vrt_file, size=size, buffer=buffer)

        if cleanup:
            for file in rows:
                os.remove(file)

        return merged

    return None


def process_rows(vrt_file: str, contours_dir: str = '.', size: int = 500, buffer: int = 10,
                 threshold: float = 0.6, cleanup: bool = True, skip_existing: bool = True,
                 max_workers: int = 4) -> Tuple[str, bool, str]:
    with rasterio.open(vrt_file) as src:
        meta = src.meta
        vrt_dim = meta['width'], meta['height']
        transform = meta['transform']

    partial_process_row = partial(_process_row, vrt_file=vrt_file, vrt_dim=vrt_dim, contours_dir=contours_dir,
                                  size=size, buff=buffer, threshold=threshold, cleanup=cleanup,
                                  skip_existing=skip_existing,
                                  transform=transform)

    rows = list(range(0, vrt_dim[1], size - buffer))

    return multiprocess(partial_process_row, rows, max_workers=max_workers)


def merging_rows(row_dict: dict, skip_existing: bool = True) -> str:
    """ merge row files into a single file per utm """
    start = time.time()
    merged_contours_file = f'{row_dict["contours_dir"]}/merged_{row_dict["time_interval"]}_{row_dict["utm"]}.gpkg'
    if skip_existing and os.path.exists(merged_contours_file):
        return merged_contours_file

    merged = merge_rows(rows=row_dict['rows'], vrt_file=row_dict['vrt_file'],
                        size=row_dict['chunk_size'], buffer=row_dict['chunk_overlap'])
    merged.to_file(merged_contours_file, driver='GPKG')

    LOGGER.info(f'Merging rows and writing results for {row_dict["time_interval"]}/{row_dict["utm"]} done'
                f' in {(time.time() - start) / 60} min!\n\n')
    return merged_contours_file


def run_vectorisation(config: VectorisationConfig) -> List[str]:
    """ Run vectorisation process on entire AOI for the given time intervals """
    filesystem = prepare_filesystem(config)

    LOGGER.info(f'Copy tiff files locally to {config.predictions_dir}')
    for time_interval in config.time_intervals:
        if not os.path.exists(f'{config.predictions_dir}/{time_interval}'):
            if not filesystem.exists(f'{config.tiffs_folder}/{time_interval}/'):
                filesystem.makedirs(f'{config.tiffs_folder}/{time_interval}/')
            copy_dir(filesystem, f'{config.tiffs_folder}/{time_interval}/',
                     f'{config.predictions_dir}/', f'{time_interval}')

    LOGGER.info(f'Move files to utm folders')
    for time_interval in config.time_intervals:
        for utm in config.utms:
            utm_dir = f'{config.predictions_dir}/{time_interval}/utm{utm}'
            os.makedirs(utm_dir, exist_ok=True)
            tiffs_to_move = glob(f'{config.predictions_dir}/{time_interval}/*-{utm}.tiff')
            for tiff in tiffs_to_move:
                tiff_name = os.path.basename(tiff)
                os.rename(tiff, f'{utm_dir}/{tiff_name}')

    LOGGER.info(f'Create weights file {config.weights_file}')
    with rasterio.open(config.weights_file, 'w', driver='gTIFF', width=config.shape[0], height=config.shape[1], count=1,
                       dtype=np.float32) as dst:
        dst.write_band(1, get_weights(config.shape, config.buffer))

    rows = []
    for time_interval in config.time_intervals:
        for utm in config.utms:
            start = time.time()
            LOGGER.info(f'Running contours for {time_interval}/{utm}!')

            contours_dir = f'{config.contours_dir}/{time_interval}/utm{utm}/'
            LOGGER.info(f'Create contour folder {contours_dir}')
            os.makedirs(contours_dir, exist_ok=True)

            predictions_dir = f'{config.predictions_dir}/{time_interval}/utm{utm}/'
            tifs = glob(f'{predictions_dir}*.tiff')
            output_vrt = f'{config.vrt_dir}/vrt_{time_interval}_{utm}.vrt'
            write_vrt(tifs, config.weights_file, output_vrt)

            results = process_rows(output_vrt, contours_dir,
                                   max_workers=config.max_workers,
                                   size=config.chunk_size,
                                   buffer=config.chunk_overlap,
                                   threshold=config.threshold,
                                   cleanup=config.cleanup,
                                   skip_existing=config.skip_existing)

            failed = [(file, excp) for file, finished, excp in results if not finished]
            if len(failed):
                LOGGER.warning('Some rows failed:')
                LOGGER.warning('\n'.join([f'{file}: {excp}' for file, excp in failed]))
                # raise Exception(f'{len(failed)} rows failed! ')
                LOGGER.warning(f'{len(failed)} rows failed! ')

            rows.append({'time_interval': time_interval,
                         'utm': utm,
                         'vrt_file': output_vrt,
                         'rows': [file for file, finished, _ in results if finished],
                         'chunk_size': config.chunk_size,
                         'chunk_overlap': config.chunk_overlap,
                         'contours_dir': config.contours_dir
                         })

            LOGGER.info(
                f'Row contours processing for {time_interval}/{utm} done in {(time.time() - start) / 60} min!\n\n')

    list_of_merged_files = multiprocess(merging_rows, rows, max_workers=config.max_workers)

    return list_of_merged_files


def utm_zone_merging(config: MergeUTMsConfig, overlap_df: gpd.GeoDataFrame, zones: gpd.GeoDataFrame):
    """
    Function to perform utm zone merging. Currently support merging of 2 UTM zones only

    It is somewhat of a concept, so the code above (getting the overlap) still has to be run before this one
    """
    assert len(config.utms) == 2, 'The function supports merging of 2 UTMs only at the moment'
    assert CRS(config.resulting_crs).pyproj_crs().axis_info[0].unit_name == 'metre', \
        'The resulting CRS should have axis units in metres.'

    for time_window in config.time_intervals:
        LOGGER.info(f'merging utms for {time_window} ...')
        merged_dfs = [gpd.read_file(f'{config.contours_dir}/merged_{time_window}_{utm}.gpkg')
                      for utm in config.utms]

        # to speed up some processing, remove the biggest fields beforehand
        LOGGER.info(f'\tfilter vectors by area ...')
        if config.max_area:
            merged_dfs = [merged_df[merged_df.geometry.area < config.max_area] for merged_df in merged_dfs]

        LOGGER.info(f'\tsplitting away non-overlapping zones ...')
        non_overlapping_utms, overlapping_utms = [], []
        for merged_df, utm in zip(merged_dfs, config.utms):
            non_over, over = split_intersecting(merged_df, overlap_df.to_crs(epsg=int(utm)).iloc[0].geometry)

            zone = zones[zones['crs'] == utm].to_crs(epsg=int(utm)).iloc[0].geometry

            over['distance'] = over.geometry.centroid.distance(zone)

            over.to_crs(epsg=4326, inplace=True)

            non_overlapping_utms.append(non_over)
            overlapping_utms.append(over)

        prefixes = ['l', 'r']

        for overlapping_utm, prefix in zip(overlapping_utms, prefixes):
            overlapping_utm[f'{prefix}_geom'] = overlapping_utm.geometry
            overlapping_utm[f'{prefix}_index'] = overlapping_utm.index

        LOGGER.info(f'\tfinding overlapping geometries with sjoin ...')
        overlaps = sjoin(overlapping_utms[0], overlapping_utms[1], how='inner', op='intersects')

        reminder_utms = [overlapping_utm[~overlapping_utm[f'{prefix}_index'].isin(
            overlaps[f'{prefix}_index'])][['geometry']].copy()
                         for overlapping_utm, prefix in zip(overlapping_utms, prefixes)]

        LOGGER.info(f'\trunning union of {len(overlaps)} overlapping geometries ...')
        overlaps['geometry'] = overlaps.apply(lambda r: r.l_geom.union(r.r_geom), axis=1)

        overlaps = overlaps[~(overlaps.is_empty | overlaps.geometry.area.isna())]

        LOGGER.info(f'\tcreate dataframe of overlaps ...')
        unified_geoms = unary_union(list(overlaps.geometry)).geoms
        merged_overlaps = gpd.GeoDataFrame(geometry=[geom for geom in unified_geoms],
                                           crs=overlaps.crs)
        merged_overlaps.to_crs(config.resulting_crs, inplace=True)

        LOGGER.info(f'\tmerging results ...')
        for gdf in non_overlapping_utms + reminder_utms:
            gdf.to_crs(config.resulting_crs, inplace=True)

        gdfs_to_merge = non_overlapping_utms + reminder_utms + [merged_overlaps]
        delineated_fields = gpd.GeoDataFrame(pd.concat(gdfs_to_merge), crs=config.resulting_crs)

        delineated_fields = delineated_fields[delineated_fields.geometry.area < config.max_area]

        LOGGER.info(f'\tsimplifying geometries ...')
        delineated_fields['geometry'] = delineated_fields.geometry.simplify(config.simplify_tolerance)

        LOGGER.info(f'\twriting output ...')
        delineated_fields.to_file(f'{config.contours_dir}/delineated_fields_{time_window}.gpkg', driver='GPKG')
