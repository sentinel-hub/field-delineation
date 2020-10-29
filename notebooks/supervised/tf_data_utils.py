import os
import numpy as np
import pandas as pd
import tensorflow as tf


class Unpack(object):
    def __call__(self, sample):
        return sample['features'], sample['labels']
    
    
class ToFloat32(object):
    def __call__(self, feats, labels):
        feats = tf.cast(feats, tf.float32)
        return feats, labels
    
    
class SetValueToNewValue(object):
    def __init__(self, old_value=2**16-1, new_value=-1):
        self.old_value = old_value
        self.new_value = new_value
        
    def __call__(self, feats, labels):
        feats = tf.where(feats==self.old_value, tf.constant(self.new_value), feats)
        return feats, labels
    
    
class OneMinusEncoding(object):
    """ encodes labels to 1-p, p """
    def __init__(self, n_classes):
        assert n_classes==2, 'OneMinus works only for "binary" classes. `n_classes` should be 2.'
        self.n_classes = n_classes
      
    def __call__(self, feats, labels):
        return feats, tf.concat([tf.ones_like(labels) - labels, labels], axis=-1)
    
    
class OneHotEncoding(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        
    def __call__(self, feats, labels):
        labels_oh = tf.one_hot(tf.squeeze(labels, axis=-1), depth=self.n_classes)
        return feats, labels_oh

    
class FillNaN(object):
    def __init__(self, fill_value=-2):
        self.fill_value = fill_value
        
    def __call__(self, feats, labels):            
        feats = tf.where(tf.math.is_nan(feats), tf.constant(self.fill_value, feats.dtype), feats)
        return feats, labels
    
    
class Normalize(object):
    def __init__(self, scaler, mean=None):
        self.scaler = scaler
        self.mean = mean
        
    def __call__(self, feats, labels):
        if self.mean is not None:
            feats = tf.math.subtract(feats, tf.convert_to_tensor(self.mean, dtype=np.float32))
        feats = tf.math.divide(feats, tf.convert_to_tensor(self.scaler, dtype=np.float32))
        return feats, labels
    
    
class LabelsToDict(object):
    def __init__(self, keys):
        self.keys = keys
        
    def __call__(self, feats, labels):
        assert len(self.keys) == labels.shape[0]
        labels_dict = {}
        for idx, key in enumerate(self.keys):
            labels_dict[key] = labels[idx,...]
        return {'features': feats}, labels_dict
    
    
def normalize_perc(ds_keys):
    feats = tf.math.divide(tf.cast(ds_keys['features'], tf.float64), ds_keys['norm_perc99'])
    ds_keys['features'] = feats
    return ds_keys


def normalize_meanstd(ds_keys):
    feats = tf.math.subtract(tf.cast(ds_keys['features'], tf.float64), ds_keys['norm_meanstd_mean'])
    feats = tf.math.divide(feats, ds_keys['norm_meanstd_std'])
    ds_keys['features'] = feats
    return ds_keys    


def augment_data(features_augmentations, labels_augmentation, brightness_delta=0.1, contrast_bounds=(0.9,1.1)):
    """ Builds a function that randomly augments features in specified ways.
    param features_to_augment: List of features to augment and which operations to perform on them.
                               Each element is of shape (feature, list_of_operations).
    type features_to_augment: list of (str, list of str)
    param brightness_delta: Maximum brightness change.
    type brightness_delta: float
    param contrast_bounds: Upper and lower bounds of contrast multiplier.
    type contrast_bounds: (float, float)
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
            'flip_left_right': lambda x: tf.cond(flip_lr_cond, lambda: tf.image.flip_left_right(x), lambda: x),
            'flip_up_down': lambda x: tf.cond(flip_ud_cond, lambda: tf.image.flip_up_down(x), lambda: x),
            'rotate': lambda x: tf.image.rot90(x, rot90_amount),
            'brightness': lambda x: tf.image.random_brightness(x, brightness_delta),
            'contrast': lambda x: tf.image.random_contrast(x, contrast_lower, contrast_upper)
        }
        

        for op in features_augmentations:
            features = _augment_data(features, operations[op])
        
        for op in labels_augmentation:
            labels = _augment_labels(labels, operations[op])
                     
        return features, labels

    return _augment


def _construct_norm_arrays(file_path):
    mode = os.path.basename(os.path.dirname(file_path))
    if mode == 'train': 
        df = pd.read_csv('../../input-data/train_dataset.csv')
    elif mode == 'test': 
        df = pd.read_csv('../../input-data/test_dataset.csv')
    elif mode == 'validation': 
        df = pd.read_csv('../../input-data/validation_dataset.csv')

    df = df[df.chunk == os.path.basename(file_path)]
    
    perc99 = df[['norm_perc99_b0', 'norm_perc99_b1', 'norm_perc99_b2', 'norm_perc99_b3']].values
    meanstd_mean = df[['norm_meanstd_mean_b0', 'norm_meanstd_mean_b1', 'norm_meanstd_mean_b2', 'norm_meanstd_mean_b3']].values
    meanstd_std = df[['norm_meanstd_std_b0', 'norm_meanstd_std_b1', 'norm_meanstd_std_b2', 'norm_meanstd_std_b3']].values
    
    return perc99, meanstd_mean, meanstd_std
        

def npz_dir_dataset(file_dir_or_list, features, randomize=True, num_parallel=5, shuffle_size=500, filesystem=None):
    """ Creates a tf.data.Dataset from a directory containing numpy .npz files. Files are loaded
    lazily when needed. `num_parallel` files are read in parallel and interleaved together.
    :param file_dir_or_list: directory containing .npz files or a list of paths to .npz files
    :type file_dir_or_list: str | list(str)
    :param features: dict of (`field` -> `feature_name`) mappings, where `field` is the field in the .npz array
                   and `feature_name` is the name of the feature it is saved to.
    :type features: dict
    :param randomize: Whether to shuffle samples returned from dataset
    :type randomize: bool, optional
    :param num_parallel: number of files to read in parallel and intereleave, defaults to 5
    :type num_parallel: int, optional
    :param shuffle_size: buffer size for shuffling file order, defaults to 500
    :type shuffle_size: int, optional
    :return: dataset containing examples merged from files
    :rtype: tf.data.Dataset
    """

    files = file_dir_or_list

    # If dir, then list files
    if isinstance(file_dir_or_list, str):
        if filesystem is None:
            dir_list = os.listdir(file_dir_or_list)
        else:
            dir_list = filesystem.listdir(file_dir_or_list)
        
        files = [os.path.join(file_dir_or_list, f) for f in dir_list]

    fields = list(features.keys())
    feature_names = [features[f] for f in features]

    # Read one file for shape info
    file = next(iter(files))

    if filesystem is None:
        data = np.load(file)
    else:
        data = np.load(filesystem.openbin(file))

    np_arrays = [data[f] for f in fields]

    # Append norm arrays 
    perc99, meanstd_mean, meanstd_std = _construct_norm_arrays(file)
    
    np_arrays.append(perc99)
    np_arrays.append(meanstd_mean)
    np_arrays.append(meanstd_std)

    # Read shape and type info
    types = tuple(arr.dtype for arr in np_arrays)
    shapes = tuple(arr.shape[1:] for arr in np_arrays)
#     print(shapes)

    # Create datasets
    datasets = [_npz_file_lazy_dataset(file, fields, feature_names, types, shapes, filesystem) for file in files]
    ds = tf.data.Dataset.from_tensor_slices(datasets)

    # Shuffle files and interleave multiple files in parallel
    if randomize:
        ds = ds.shuffle(shuffle_size)
    
    ds = ds.interleave(lambda x:x, cycle_length=num_parallel)

    return ds


def _npz_file_lazy_dataset(file_path, fields, feature_names, types, shapes, filesystem=None):
    """ Creates a lazy tf.data Dataset from a numpy file.
    Reads the file when first consumed.
    :param file_path: path to the numpy file
    :type file_path: str
    :param fields: fields to read from the numpy file
    :type fields: list(str)
    :param feature_names: feature names assigned to the fields
    :type feature_names: list(str)
    :param types: types of the numpy fields
    :type types: list(np.dtype)
    :param shapes: shapes of the numpy fields
    :type shapes: list(tuple)
    :return: dataset containing examples from the file
    :rtype: tf.data.Dataset
    """

    def _generator():
        if filesystem is None:
            data = np.load(file_path)
        else:
            data = np.load(filesystem.openbin(file_path))

        np_arrays = [data[f] for f in fields]
        
        perc99, meanstd_mean, meanstd_std = _construct_norm_arrays(file_path)

        np_arrays.append(perc99)
        np_arrays.append(meanstd_mean)
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
                'norm_meanstd_mean': [features[5]], 
                'norm_meanstd_std': features[6]}

    ds = ds.map(_to_dict)

    return ds

