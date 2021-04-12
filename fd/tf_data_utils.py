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
from functools import partial

from typing import List, Tuple, Callable, Union

from fs_s3fs import S3FS

import numpy as np
import pandas as pd
import tensorflow as tf


class Unpack(object):
    """ Unpack items of a dictionary to a tuple """
    def __call__(self, sample: dict) -> Tuple[tf.Tensor, tf.Tensor]:
        return sample['features'], sample['labels']
    
    
class ToFloat32(object):
    """ Cast features to float32 """
    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        feats = tf.cast(feats, tf.float32)
        return feats, labels
    
    
class SetValueToNewValue(object):
    """ Set a value in the features array """
    def __init__(self, old_value: float = 2**16-1, new_value: float = -1):
        self.old_value = old_value
        self.new_value = new_value
        
    def __call__(self, feats, labels) -> Tuple[tf.Tensor, tf.Tensor]:
        feats = tf.where(feats == self.old_value, tf.constant(self.new_value), feats)
        return feats, labels
    
    
class OneMinusEncoding(object):
    """ Encodes labels to 1-p, p. Makes sense only for binary labels and for continuous labels in [0, 1] """
    def __init__(self, n_classes: int):
        assert n_classes == 2, 'OneMinus works only for "binary" classes. `n_classes` should be 2.'
        self.n_classes = n_classes
      
    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return feats, tf.concat([tf.ones_like(labels) - labels, labels], axis=-1)
    
    
class OneHotEncoding(object):
    """ One hot encoding of categorical labels """
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        
    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        labels_oh = tf.one_hot(tf.squeeze(labels, axis=-1), depth=self.n_classes)
        return feats, labels_oh

    
class FillNaN(object):
    """ Replace NaN values with a given finite value """
    def __init__(self, fill_value: float = -2.):
        self.fill_value = fill_value
        
    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        feats = tf.where(tf.math.is_nan(feats), tf.constant(self.fill_value, feats.dtype), feats)
        return feats, labels
    
    
class Normalize(object):
    """ Apply normalization to the features """
    def __init__(self, scaler: float, mean: float = None):
        self.scaler = scaler
        self.mean = mean
        
    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.mean is not None:
            feats = tf.math.subtract(feats, tf.convert_to_tensor(self.mean, dtype=np.float32))
        feats = tf.math.divide(feats, tf.convert_to_tensor(self.scaler, dtype=np.float32))
        return feats, labels
    
    
class LabelsToDict(object):
    """ Convert a list of arrays to a dictionary """
    def __init__(self, keys: List[str]):
        self.keys = keys
        
    def __call__(self, feats: tf.Tensor, labels: tf.Tensor) -> Tuple[dict, dict]:
        assert len(self.keys) == labels.shape[0]
        labels_dict = {}
        for idx, key in enumerate(self.keys):
            labels_dict[key] = labels[idx, ...]
        return {'features': feats}, labels_dict
    
    
def normalize_perc(ds_keys: dict) -> dict:
    """ Help function to normalise the features by the 99th percentile """
    feats = tf.math.divide(tf.cast(ds_keys['features'], tf.float64), ds_keys['norm_perc99'])
    ds_keys['features'] = feats
    return ds_keys


def normalize_meanstd(ds_keys: dict, subtract: str = 'mean') -> dict:
    """ Help function to normalise the features by the mean and standard deviation """
    assert subtract in ['mean', 'median']
    feats = tf.math.subtract(tf.cast(ds_keys['features'], tf.float64), ds_keys[f'norm_meanstd_{subtract}'])
    feats = tf.math.divide(feats, ds_keys['norm_meanstd_std'])
    ds_keys['features'] = feats
    return ds_keys


def flip_left_right(x: tf.Tensor, flip_lr_cond: bool = False) -> tf.Tensor:
    if flip_lr_cond:
        return tf.image.flip_left_right(x)
    return x


def flip_up_down(x: tf.Tensor, flip_ud_cond: bool = False) -> tf.Tensor:
    if flip_ud_cond:
        return tf.image.flip_up_down(x)
    return x


def rotate(x: tf.Tensor, rot90_amount: int = 0) -> tf.Tensor:
    return tf.image.rot90(x, rot90_amount)


def brightness(x: tf.Tensor, brightness_delta: float = .0) -> tf.Tensor:
    return tf.image.random_brightness(x, brightness_delta)


def contrast(x: tf.Tensor, contrast_lower: float = .9, contrast_upper=1.1) -> tf.Tensor:
    return tf.image.random_contrast(x, contrast_lower, contrast_upper)


def augment_data(features_augmentations: List[str],
                 labels_augmentation: List[str],
                 brightness_delta: float = 0.1, contrast_bounds: Tuple[float, float] = (0.9, 1.1)) -> Callable:
    """ Builds a function that randomly augments features in specified ways.

    param features_to_augment: List of features to augment and which operations to perform on them.
                               Each element is of shape (feature, list_of_operations).
    param brightness_delta: Maximum brightness change.
    param contrast_bounds: Upper and lower bounds of contrast multiplier.
    """
    def _augment_data(data, op_fn):
        return op_fn(data)
    
    def _augment_labels(labels_augmented, oper_op):
        ys = [] 
        for i in range(len(labels_augmented)): 
            ys.append(_augment_data(labels_augmented[i, ...], oper_op))
        return tf.convert_to_tensor(ys, dtype=labels_augmented.dtype)

    def _augment(features, labels):
        contrast_lower, contrast_upper = contrast_bounds

        flip_lr_cond = tf.random.uniform(shape=[]) > 0.5
        flip_ud_cond = tf.random.uniform(shape=[]) > 0.5
        rot90_amount = tf.random.uniform(shape=[], maxval=4, dtype=tf.int32)

        # Available operations
        operations = {
            'flip_left_right': partial(flip_left_right, flip_lr_cond=flip_lr_cond),
            'flip_up_down': partial(flip_up_down, flip_ud_cond=flip_ud_cond),
            'rotate': partial(rotate, rot90_amount=rot90_amount),
            'brightness': partial(brightness, brightness_delta=brightness_delta),
            'contrast': partial(contrast, contrast_lower=contrast_lower, contrast_upper=contrast_upper)
        }

        for op in features_augmentations:
            features = _augment_data(features, operations[op])
        
        for op in labels_augmentation:
            labels = _augment_labels(labels, operations[op])
                     
        return features, labels

    return _augment


def _construct_norm_arrays(file_path: str, metadata_path: str, fold: int = None, filesystem: S3FS = None) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Return arrays with normalisation factors to be used """
    chunk_name = os.path.basename(file_path)

    df = pd.read_csv(filesystem.open(metadata_path)) if filesystem is not None else pd.read_csv(metadata_path)

    df = df[df.chunk == chunk_name]

    if fold is not None:
        df = df[df.fold == fold]

    perc99 = df[['norm_perc99_b0', 'norm_perc99_b1', 'norm_perc99_b2', 'norm_perc99_b3']].values
    meanstd_mean = df[['norm_meanstd_mean_b0', 'norm_meanstd_mean_b1',
                       'norm_meanstd_mean_b2', 'norm_meanstd_mean_b3']].values
    meanstd_median = df[['norm_meanstd_median_b0', 'norm_meanstd_median_b1',
                         'norm_meanstd_median_b2', 'norm_meanstd_median_b3']].values
    meanstd_std = df[['norm_meanstd_std_b0', 'norm_meanstd_std_b1',
                      'norm_meanstd_std_b2', 'norm_meanstd_std_b3']].values
    
    return perc99, meanstd_mean, meanstd_median, meanstd_std
        

def npz_dir_dataset(file_dir_or_list: Union[str, List[str]], features: dict, metadata_path: str,
                    fold: int = None, randomize: bool = True,
                    num_parallel: int = 5, shuffle_size: int = 500,
                    filesystem: S3FS = None, npz_from_s3: bool = False) -> tf.data.Dataset:
    """ Creates a tf.data.Dataset from a directory containing numpy .npz files.

    Files are loaded lazily when needed. `num_parallel` files are read in parallel and interleaved together.

    :param file_dir_or_list: directory containing .npz files or a list of paths to .npz files
    :param features: dict of (`field` -> `feature_name`) mappings, where `field` is the field in the .npz array
                   and `feature_name` is the name of the feature it is saved to.
    :param fold: in k-fold validation, fold to consider when querying the patchlet info dataframe
    :param randomize: whether to shuffle the samples of the dataset or not, defaults to `True`
    :param num_parallel: number of files to read in parallel and intereleave, defaults to 5
    :param shuffle_size: buffer size for shuffling file order, defaults to 500
    :param metadata_path: path to input csv files with patchlet information
    :param filesystem: filesystem to access bucket, defaults to None
    :param npz_from_s3: if True, npz files are loaded from S3 bucket, otherwise from local disk
    :return: dataset containing examples merged from files
    """

    files = file_dir_or_list

    if npz_from_s3:
        assert filesystem is not None
    
    # If dir, then list files
    if isinstance(file_dir_or_list, str):
        if filesystem and not filesystem.isdir(file_dir_or_list):
            filesystem.makedirs(file_dir_or_list)
        dir_list = os.listdir(file_dir_or_list) if not npz_from_s3 else filesystem.listdir(file_dir_or_list)
        files = [os.path.join(file_dir_or_list, f) for f in dir_list]
        
    fields = list(features.keys())

    # Read one file for shape info
    file = next(iter(files))
    data = np.load(file) if not npz_from_s3 else np.load(filesystem.openbin(file))
    np_arrays = [data[f] for f in fields]

    # Append norm arrays 
    perc99, meanstd_mean, meanstd_median, meanstd_std = _construct_norm_arrays(file, metadata_path, fold, filesystem)
    
    np_arrays.append(perc99)
    np_arrays.append(meanstd_mean)
    np_arrays.append(meanstd_median)
    np_arrays.append(meanstd_std)

    # Read shape and type info
#     types = tuple(arr.dtype for arr in np_arrays)
    types = (tf.uint16, tf.float32, tf.float32, tf.float32, tf.float64, tf.float64, tf.float64, tf.float64)
    shapes = tuple(arr.shape[1:] for arr in np_arrays)

    # Create datasets
    datasets = [_npz_file_lazy_dataset(file, fields, types, shapes, metadata_path, fold=fold,
                                       filesystem=filesystem, npz_from_s3=npz_from_s3) for file in files]
    ds = tf.data.Dataset.from_tensor_slices(datasets)

    # Shuffle files and interleave multiple files in parallel
    if randomize:
        ds = ds.shuffle(shuffle_size)
    
    ds = ds.interleave(lambda x: x, cycle_length=num_parallel)

    return ds


def _npz_file_lazy_dataset(file_path: str, fields: List[str], types: List[np.dtype], shapes: List[np.int],
                           metadata_path: str, fold: int = None, filesystem: S3FS = None, 
                           npz_from_s3: bool = False) -> tf.data.Dataset:
    """ Creates a lazy tf.data Dataset from a numpy file.

    Reads the file when first consumed.

    :param file_path: path to the numpy file
    :param fields: fields to read from the numpy file
    :param types: types of the numpy fields
    :param shapes: shapes of the numpy fields
    :param metadata_path: path to csv files with patchlet info
    :param fold: number of cross-validation fold to consider
    :param filesystem: S3 filesystem, defaults to None
    :param npz_from_s3: whether to load npz files from bucket if True, or from local disk if False
    :return: dataset containing examples from the file
    """

    def _generator():
        data = np.load(file_path) if not npz_from_s3 else np.load(filesystem.openbin(file_path))
        np_arrays = [data[f] for f in fields]
        
        perc99, meanstd_mean, meanstd_median, meanstd_std = _construct_norm_arrays(file_path, metadata_path,
                                                                                   fold, filesystem)

        np_arrays.append(perc99)
        np_arrays.append(meanstd_mean)
        np_arrays.append(meanstd_median)
        np_arrays.append(meanstd_std)
        
        # Check that arrays match in the first dimension
        n_samples = np_arrays[0].shape[0]
        assert all(n_samples == arr.shape[0] for arr in np_arrays)
        # Iterate through the first dimension of arrays
        for slices in zip(*np_arrays):
            yield slices

    ds = tf.data.Dataset.from_generator(_generator, types, shapes)

    # Converts a database of tuples to database of dicts
    def _to_dict(*features):
        return {'features': features[0], 
                'labels': [features[1], features[2], features[3]], 
                'norm_perc99': features[4], 
                'norm_meanstd_mean': features[5],
                'norm_meanstd_median': features[6],
                'norm_meanstd_std': features[7]}

    ds = ds.map(_to_dict)

    return ds

