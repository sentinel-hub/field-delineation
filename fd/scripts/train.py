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
import sys
import json
import logging
import argparse
from datetime import datetime
from functools import reduce

import numpy as np
import tensorflow as tf
from fs.copy import copy_dir
from tqdm.auto import tqdm 

from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.losses import TanimotoDistanceLoss

from eoflow.models.segmentation_unets import ResUnetA

from fd.training import TrainingConfig, get_dataset, initialise_model, initialise_callbacks
from fd.utils import prepare_filesystem, LogFileFilter


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


def train_k_folds(config: dict):
    """ Utility function to create K tf-datasets and train k-models

    Args:
        config (dict): Config dictionary with k-fold training parameters
    """

    training_config = TrainingConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config['aws_access_key_id'], 
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'],
        wandb_id=config['wandb_id'], 
        npz_folder=config['npz_folder'],
        metadata_path=config['metadata_path'],
        model_folder=config['model_folder'],
        model_s3_folder=config['model_s3_folder'],
        chkpt_folder=config['chkpt_folder'],
        input_shape=tuple(config['input_shape']),
        n_classes=config['n_classes'],
        batch_size=config['batch_size'],
        iterations_per_epoch=config['iterations_per_epoch'], 
        num_epochs=config['num_epochs'],
        model_name=config['model_name'],
        reference_names=config['reference_names'],
        augmentations_feature=config['augmentations_feature'],
        augmentations_label=config['augmentations_label'],
        normalize=config['normalize'],
        n_folds=config['n_folds'],
        model_config=config['model_config'],
        fill_value=config['fill_value'],
        seed=config['seed']
    )

    if training_config.wandb_id is not None:
        os.system(f'wandb login {training_config.wandb_id}')

    LOGGER.info('Create K TF datasets')
    ds_folds = [get_dataset(training_config, fold=fold, augment=True, randomize=True,
                            num_parallel=config['num_parallel'], npz_from_s3=config['npz_from_s3']) 
                for fold in tqdm(range(1, training_config.n_folds+1))]

    folds = list(range(training_config.n_folds))

    folds_ids_list = [(folds[:nf] + folds[1 + nf:], [nf]) for nf in folds]

    np.random.seed(training_config.seed)

    models = []
    model_paths = []

    for training_ids, testing_id in folds_ids_list:
        
        left_out_fold = testing_id[0]+1
        LOGGER.info(f'Training model for left-out fold {left_out_fold}')
        
        fold_val = np.random.choice(training_ids)
        folds_train = [tid for tid in training_ids if tid != fold_val]
        LOGGER.info(f'\tTrain folds {folds_train}, Val fold: {fold_val}, Test fold: {testing_id[0]}')
        
        ds_folds_train = [ds_folds[tid] for tid in folds_train]
        ds_train = reduce(tf.data.Dataset.concatenate, ds_folds_train)
            
        ds_val = ds_folds[fold_val]
        ds_val = ds_val.batch(training_config.batch_size)
        
        ds_train = ds_train.batch(training_config.batch_size)
        ds_train = ds_train.repeat()
        
        # Get model
        model = initialise_model(training_config, chkpt_folder=training_config.chkpt_folder)
        
        # Set up callbacks to monitor training
        model_path, callbacks = initialise_callbacks(training_config, 
                                                    fold=left_out_fold)
        
        LOGGER.info(f'\tTraining model, writing to {model_path}')
        
        model.net.fit(ds_train, 
                      validation_data=ds_val,
                      epochs=training_config.num_epochs,
                      steps_per_epoch=training_config.iterations_per_epoch,
                      callbacks=callbacks)
        
        models.append(model)
        model_paths.append(model_path)
        
        del fold_val, folds_train, ds_train, ds_val, ds_folds_train

    LOGGER.info('Copy model directories to bucket')
    filesystem = prepare_filesystem(training_config)

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        filesystem.makedirs(f'{training_config.model_s3_folder}/{model_name}', recreate=True)
        copy_dir(training_config.model_folder, 
                f'{model_name}',
                filesystem, 
                f'{training_config.model_s3_folder}/{model_name}')

    LOGGER.info('Create average model')
    weights = [model.net.get_weights() for model in models]

    avg_weights = list()
    for weights_list_tuple in zip(*weights):
        avg_weights.append(np.array([np.array(weights_).mean(axis=0) 
                            for weights_ in zip(*weights_list_tuple)]))

    avg_model = initialise_model(training_config)
    avg_model.net.set_weights(avg_weights)

    now = datetime.now().isoformat(sep='-', timespec='seconds').replace(':', '-')
    model_path = f'{training_config.model_folder}/{training_config.model_name}_avg_{now}'

    LOGGER.info('Save average model to local path')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')
    with open(f'{model_path}/model_cfg.json', 'w') as jfile:
        json.dump(training_config.model_config, jfile)
    avg_model.net.save_weights(checkpoints_path)

    for _, testing_id in folds_ids_list:
        
        left_out_fold = testing_id[0]+1

        LOGGER.info(f'Evaluating model on left-out fold {left_out_fold}')
        model = models[testing_id[0]]
        model.net.evaluate(ds_folds[testing_id[0]].batch(training_config.batch_size))

        LOGGER.info(f'Evaluating average model on left-out fold {left_out_fold}')
        avg_model.net.evaluate(ds_folds[testing_id[0]].batch(training_config.batch_size))
        LOGGER.info('\n\n')


if __name__ == '__main__':
    LOGGER.info(f'Reading configuration from {args.config}')

    parser = argparse.ArgumentParser(description="Train models in a k-fold cross-validation.\n")

    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file with k-fold training parameters", 
        required=True
    )
    args = parser.parse_args()


    with open(args.config, 'r') as jfile:
        cfg_dict = json.load(jfile)

    train_k_folds(cfg_dict)
