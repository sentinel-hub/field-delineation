# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from eoflow.models.losses import TanimotoDistanceLoss
from eoflow.models.metrics import MCCMetric
from eoflow.models.segmentation_unets import ResUnetA
from eolearn.core import EOPatch, FeatureType, LoadTask, OverwritePermission, SaveTask
from fs.copy import copy_dir, copy_file
from fs_s3fs import S3FS
from tensorflow.keras.metrics import CategoricalAccuracy, MeanIoU

from .utils import BaseConfig, set_sh_config


@dataclass
class PredictionConfig(BaseConfig):
    eopatches_folder: str
    feature_extent: Tuple[FeatureType, str]
    feature_boundary: Tuple[FeatureType, str]
    feature_distance: Tuple[FeatureType, str]
    model_path: str
    model_name: str
    model_version: str
    temp_model_path: str
    height: int
    width: int
    n_channels: int
    n_classes: int
    metadata_path: str
    batch_size: int
    normalise: str
    pad_buffer: int
    crop_buffer: int
    reference_extent: Optional[Tuple[FeatureType, str]] = None 
    reference_boundary: Optional[Tuple[FeatureType, str]] = None 
    reference_distance: Optional[Tuple[FeatureType, str]] = None
    feature_bands: Tuple[FeatureType, str] = (FeatureType.DATA, 'BANDS')


def binary_one_hot_encoder(array: np.ndarray) -> np.ndarray:
    """ One hot encode the label array along the last dimension """
    return np.concatenate([1 - array, array], axis=-1)


def crop_array(array: np.ndarray, buffer: int) -> np.ndarray:
    """ Crop height and width of a 4D array given a buffer size. Array has shape B x H x W x C """
    assert array.ndim == 4, 'Input array of wrong dimension, needs to be 4D B x H x W x C'

    return array[:, buffer:-buffer:, buffer:-buffer:, :]


def pad_array(array: np.ndarray, buffer: int) -> np.ndarray:
    """ Pad height and width dimensions of a 4D array with a given buffer. Height and with are in 2nd and 3rd dim """
    assert array.ndim == 4, 'Input array of wrong dimension, needs to be 4D B x H x W x C'

    return np.pad(array, [(0, 0), (buffer, buffer), (buffer, buffer), (0, 0)], mode='edge')


def get_tanimoto_loss(from_logits: bool = False) -> TanimotoDistanceLoss:
    return TanimotoDistanceLoss(from_logits=from_logits)


def get_accuracy_metric(name: str = 'accuracy') -> CategoricalAccuracy:
    return CategoricalAccuracy(name=name)


def get_iou_metric(n_classes: int, name: str = 'iou') -> MeanIoU:
    return MeanIoU(num_classes=n_classes, name=name)


def get_mcc_metric(n_classes: int, threshold: float = .5) -> MCCMetric:
    mcc_metric = MCCMetric(default_n_classes=n_classes, default_threshold=threshold)
    mcc_metric.init_from_config({'n_classes': n_classes})
    return mcc_metric


def prediction_fn(eop: EOPatch, n_classes: int,
                  normalisation_factors: pd.DataFrame,
                  normalise: str,
                  model: ResUnetA, model_name: str,
                  extent_feature: Tuple[FeatureType, str],
                  boundary_feature: Tuple[FeatureType, str],
                  distance_feature: Tuple[FeatureType, str],
                  suffix: str,
                  batch_size: int,
                  bands_feature: Tuple[FeatureType, str],
                  crop_buffer: int,
                  pad_buffer: int,
                  reference_extent: Optional[Tuple[FeatureType, str]],
                  reference_boundary:  Optional[Tuple[FeatureType, str]],
                  reference_distance:  Optional[Tuple[FeatureType, str]]) -> EOPatch:
    """ Perform prediction for all timestamps in an EOPatch given a model and normalisation factors """
    assert normalise in ['to_meanstd', 'to_medianstd']

    extent_pred, boundary_pred, distance_pred = [], [], []
    metrics = []

    padded = pad_array(eop[bands_feature], buffer=pad_buffer)

    calc_metrics = all([ref for ref in [reference_extent, reference_distance, reference_boundary]])

    if calc_metrics:
        tanimoto_loss = get_tanimoto_loss()
        accuracy_metric = get_accuracy_metric()
        iou_metric = get_iou_metric(n_classes=n_classes)
        mcc_metric = get_mcc_metric(n_classes=n_classes)

    for timestamp, bands in zip(eop.timestamp, padded):

        month = timestamp.month

        norm_factors_month = normalisation_factors[normalisation_factors['month'] == month].iloc[0]

        if normalise == 'to_meanstd':
            avg_stat = np.array([norm_factors_month.norm_meanstd_mean_b0,
                                 norm_factors_month.norm_meanstd_mean_b1,
                                 norm_factors_month.norm_meanstd_mean_b2,
                                 norm_factors_month.norm_meanstd_mean_b3])
        else:
            avg_stat = np.array([norm_factors_month.norm_meanstd_median_b0,
                                 norm_factors_month.norm_meanstd_median_b1,
                                 norm_factors_month.norm_meanstd_median_b2,
                                 norm_factors_month.norm_meanstd_median_b3])

        dn_std = np.array([norm_factors_month.norm_meanstd_std_b0,
                           norm_factors_month.norm_meanstd_std_b1,
                           norm_factors_month.norm_meanstd_std_b2,
                           norm_factors_month.norm_meanstd_std_b3])

        data = (bands - avg_stat) / dn_std

        extent, boundary, distance = model.net.predict(data[np.newaxis, ...], batch_size=batch_size)

        extent = crop_array(extent, buffer=crop_buffer)
        boundary = crop_array(boundary, buffer=crop_buffer)
        distance = crop_array(distance, buffer=crop_buffer)

        extent_pred.append(extent)
        boundary_pred.append(boundary)
        distance_pred.append(distance)

        tmp = {}
        if calc_metrics:
            for mask_name, gt, pred in [('extent', eop[reference_extent], extent),
                                        ('boundary', eop[reference_boundary], boundary),
                                        ('distance', eop[reference_distance], distance)]:
                tmp[f'{mask_name}_loss'] = tanimoto_loss(binary_one_hot_encoder(gt[np.newaxis, ...]), pred).numpy()
                tmp[f'{mask_name}_acc'] = accuracy_metric(binary_one_hot_encoder(gt[np.newaxis, ...]), pred).numpy()
                tmp[f'{mask_name}_iou'] = iou_metric(binary_one_hot_encoder(gt[np.newaxis, ...]), pred).numpy()
                tmp[f'{mask_name}_mcc'] = mcc_metric(binary_one_hot_encoder(gt[np.newaxis, ...]), pred).numpy()[1]

                accuracy_metric.reset_states()
                iou_metric.reset_states()
                mcc_metric.reset_states()

            metrics.append(tmp)
        
        del data, extent, boundary, distance
        
    if len(extent_pred) != len(eop.timestamp):
        raise ValueError(f'Error in prediction: not all timeframes have been predicted')

    extent_pred = np.concatenate(extent_pred, axis=0)
    boundary_pred = np.concatenate(boundary_pred, axis=0)
    distance_pred = np.concatenate(distance_pred, axis=0)

    eop[extent_feature] = extent_pred[..., [1]]
    eop[boundary_feature] = boundary_pred[..., [1]]
    eop[distance_feature] = distance_pred[..., [1]]

    if calc_metrics:
        eop.meta_info[f'metrics_{suffix}'] = metrics
        eop.meta_info[f'model_{suffix}'] = model_name
    
    del extent_pred, boundary_pred, distance_pred, padded

    return eop


def load_metadata(filesystem: S3FS, config: PredictionConfig) -> pd.DataFrame:
    """ Load DataFrame with info about normalisation factors """
    metadata_dir = os.path.dirname(config.metadata_path)
    if not filesystem.exists(metadata_dir):
        filesystem.makedirs(metadata_dir)

    df = pd.read_csv(filesystem.open(f'{config.metadata_path}'))

    normalisation_factors = df.groupby(pd.to_datetime(df.timestamp).dt.to_period("M")).max()

    normalisation_factors['month'] = pd.to_datetime(normalisation_factors.timestamp).dt.month

    return normalisation_factors


def load_model(filesystem: S3FS, config: PredictionConfig) -> ResUnetA:
    """ Copy the model locally if not existing and load it """
    if not os.path.exists(f'{config.temp_model_path}/{config.model_name}'):
        if not filesystem.exists(f'{config.model_path}/{config.model_name}/checkpoints/'):
            filesystem.makedirs(f'{config.model_path}/{config.model_name}/checkpoints/')
        copy_dir(filesystem, f'{config.model_path}/{config.model_name}/checkpoints/',
                 f'{config.temp_model_path}/{config.model_name}', 'checkpoints')
        copy_file(filesystem, f'{config.model_path}/{config.model_name}/model_cfg.json',
                  f'{config.temp_model_path}/{config.model_name}', 'model_cfg.json')

    input_shape = dict(features=[None, config.height, config.width, config.n_channels])

    with open(f'{config.temp_model_path}/{config.model_name}/model_cfg.json', 'r') as jfile:
        model_cfg = json.load(jfile)

    # initialise model from config, build, compile and load trained weights
    model = ResUnetA(model_cfg)
    model.build(input_shape)
    model.net.compile()
    model.net.load_weights(f'{config.temp_model_path}/{config.model_name}/checkpoints/model.ckpt')

    return model


def run_prediction_on_eopatch(eopatch_name: str, config: PredictionConfig, filesystem: S3FS, 
                              model: ResUnetA = None, normalisation_factors: pd.DataFrame = None) -> dict:
    """ Run prediction workflow on one eopatch. Model and dataframe can be provided to avoid loading them every time """
    sh_config = set_sh_config(config)
    if normalisation_factors is None:
        normalisation_factors = load_metadata(filesystem, config)

    if model is None:
        model = load_model(filesystem, config)

    load_ref = all([ref for ref in [config.reference_distance, config.reference_extent, config.reference_boundary]])
    if load_ref:
        feats = [config.feature_bands, config.reference_distance, config.reference_extent, config.reference_boundary,
                 FeatureType.TIMESTAMP, FeatureType.META_INFO, FeatureType.BBOX]
    else:
        feats = [config.feature_bands, FeatureType.TIMESTAMP, FeatureType.META_INFO, FeatureType.BBOX]

    load_task = LoadTask(path=f's3://{config.bucket_name}/{config.eopatches_folder}',
                         features=feats,
                         config=sh_config)

    save_task = SaveTask(path=f's3://{config.bucket_name}/{config.eopatches_folder}',
                         features=[config.feature_extent, config.feature_boundary, config.feature_distance,
                                   FeatureType.META_INFO],
                         overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
                         config=sh_config)

    try:
        eop = load_task.execute(eopatch_folder=eopatch_name)

        eop = prediction_fn(eop,
                            normalisation_factors=normalisation_factors,
                            normalise=config.normalise,
                            model=model, model_name=config.model_name,
                            extent_feature=config.feature_extent,
                            boundary_feature=config.feature_boundary,
                            distance_feature=config.feature_distance,
                            suffix=config.model_version,
                            batch_size=config.batch_size,
                            n_classes=config.n_classes,
                            bands_feature=config.feature_bands,
                            pad_buffer=config.pad_buffer, 
                            crop_buffer=config.crop_buffer,
                            reference_boundary=config.reference_boundary,
                            reference_distance=config.reference_distance,
                            reference_extent=config.reference_extent)

        _ = save_task.execute(eop, eopatch_folder=eopatch_name)

        del eop, load_task, save_task

        return dict(name=eopatch_name, status='Success')

    except Exception as exc:
        return dict(name=eopatch_name, status=exc)
