#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

from typing import List, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches, patheffects

from eolearn.core import EOPatch
from shapely.geometry import Polygon, MultiPolygon

from logging import Logger
logger = Logger(__file__)


def get_extent(eopatch: EOPatch) -> Tuple[float, float, float, float]:
    """
    Calculate the extent (bounds) of the patch.

    Parameters
    ----------
    eopatch: EOPatch for which the extent is calculated.

    Returns The list of EOPatch bounds (min_x, max_x, min_y, max_y)
    -------
    """
    return eopatch.bbox.min_x, eopatch.bbox.max_x, eopatch.bbox.min_y, eopatch.bbox.max_y


def draw_true_color(ax: plt.axes, eopatch: EOPatch, time_idx: Union[List[int], int],
                    feature_name='BANDS-S2-L2A',
                    bands: Tuple[int] = (3, 2, 1),
                    factor: int = 3.5,
                    grid: bool = True):
    """
    Visualization of the bands in the EOPatch.
    Parameters
    ----------
    ax: Axis on which to plot
    eopatch: EOPatch to visualize.
    time_idx: Single timestamp or multiple timestamps.
    feature_name: Name of the feature to visualize.
    bands: Order of the bands.
    factor: Rescaling factor to
    grid: Show grid on visualization

    Returns None
    -------

    """
    def visualize_single_idx(axis, ts):
        axis.imshow(np.clip(eopatch.data[feature_name][ts][..., bands] * factor, 0, 1), extent=get_extent(eopatch))
        if grid:
            axis.grid()
            axis.set_title(f'{feature_name} {eopatch.timestamp[time_idx]}')

    if isinstance(time_idx, int):
        time_idx = [time_idx]
    if len(time_idx) == 1:
        visualize_single_idx(ax, time_idx[0])
    else:
        for i, tidx in enumerate(time_idx):
            visualize_single_idx(ax[i], tidx)


def plot_s2_eopatch(eop: EOPatch, timestamp: Union[List[int], int], data_key: str,
                    figsize: Tuple[int, int] = (10, 10)) -> None:
    """
    True color visualization of single or multiple timestamps in the EOPatch.
    Parameters
    ----------
    eop: EOPatch to visualize.
    timestamp: Single timestamp or a list of timestamps to visualize.
    data_key: Key of the data that we wish to visualize.
    figsize: Figure size of the resulting visualization.

    Returns None
    -------

    """

    if isinstance(timestamp, int):
        timestamp = [timestamp]

    fig, ax = plt.subplots(len(timestamp), figsize=figsize)
    if len(timestamp) == 1:
        ax.imshow(eop.data[data_key][timestamp[0]][..., [3, 2, 1]].squeeze()*2.5)
    else:
        for idx, ts in enumerate(timestamp):
            ax[idx].imshow(eop.data[data_key][ts][..., [3, 2, 1]].squeeze()*2.5)


def draw_outline(o, lw, foreground='black'):
    """
    Adds outline to the matplotlib patch.

    Parameters
    ----------
    o:
    lw: Linewidth
    foreground

    Returns
    -------
    """
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground=foreground), patheffects.Normal()])


def draw_poly(ax, poly: Union[Polygon, MultiPolygon], color: str = 'r', lw: int = 2, outline: bool = True):
    """
    Draws a polygon or multipolygon onto an axes.

    Parameters
    ----------
    ax: Matplotlib Axes on which to plot on
    poly: Polygon or Multipolygons to plot
    color: Color of the plotted polygon
    lw: Line width of the plot
    outline: Should the polygon be outlined

    Returns None
    -------

    """
    if isinstance(poly, MultiPolygon):
        polys = list(poly)
    else:
        polys = [poly]
    for poly in polys:
        if poly is None:
            logger.warning("One of the polygons is None.")
            break
        if poly.exterior is None:
            logger.warning("One of the polygons has not exterior.")
            break

        x, y = poly.exterior.coords.xy
        xy = np.moveaxis(np.array([x, y]), 0, -1)
        patch = ax.add_patch(patches.Polygon(xy, closed=True, edgecolor=color, fill=False, lw=lw))

    if outline:
        draw_outline(patch, 4)


def draw_bbox(ax, eopatch: EOPatch, color: str = 'r', lw: int = 2, outline: bool = True):
    """
    Plots an EOPatch bounding box onto a matplotlib axes.
    Parameters
    ----------
    ax: Matplotlib axes on which to plot.
    eopatch: EOPatch with BBOx
    color: Color of the BBOX plot.
    lw: Line width.
    outline: Should the plot be additionally outlined.

    Returns None
    -------

    """
    bbox_poly = eopatch.bbox.get_polygon()
    draw_poly(ax, Polygon(bbox_poly), color=color, lw=lw, outline=outline)


def draw_mask(ax, eopatch: EOPatch, time_idx: Union[List[int], int, None], feature_name: str, grid: bool = True,
              vmin: int = 0, vmax: int = 1, alpha: float = 1.0):
    """
    Draws an EOPatch mask or mask_timeless feature.
    Parameters
    ----------
    ax: Matplotlib axes on which to plot on
    eopatch: EOPatch for which to plot the mask:
    time_idx: Time index of the mask. If int, single time index of the mask feature, if List[int] multiple masks for
    each time index. If None, plot mask_timeless.
    feature_name: Name of the feature to plot.
    grid: Show grid on plot:
    vmin: Minimum value (for mask visualization)
    vmax: Maximum value (for mask visualization)
    alpha: Transparency of the mask
    Returns
    -------

    """

    def _show_single_ts(axis, msk, ts):
        axis.imshow(msk, extent=get_extent(eopatch), vmin=vmin, vmax=vmax, alpha=alpha)
        if grid:
            axis.grid()
        title = f'{feature_name} {eopatch.timestamp[ts]}' if ts is not None else f'{feature_name}'
        axis.set_title(title)

    if time_idx is None:
        mask = eopatch.mask_timeless[feature_name].squeeze()
        _show_single_ts(ax, mask, time_idx)
    elif isinstance(time_idx, int):
        mask = eopatch.mask[feature_name][time_idx].squeeze()
        _show_single_ts(ax, mask, time_idx)
    elif isinstance(time_idx, list):
        for i, tidx in enumerate(time_idx):
            mask = eopatch.mask[feature_name][tidx].squeeze()
            _show_single_ts(ax[i], mask, tidx)


def draw_vector_timeless(ax, eopatch: EOPatch, vector_name: str, color: str = 'b', alpha: int = 0.5):
    """
    Draws all polygons from EOPatch' timeless vector geopandas data frame.

    Parameters
    ----------
    ax: Axes on which to plot on
    eopatch: EOPatch from which to plot the vector_timeless features
    vector_name: Name of the vector_timeless feature
    color: Color of the polygons on the plot
    alpha: Transparency of the polygon on the plot.

    Returns
    -------

    """
    eopatch.vector_timeless[vector_name].plot(ax=ax, color=color, alpha=alpha)
