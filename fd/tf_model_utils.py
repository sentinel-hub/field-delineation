#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import json
import numpy as np
import tensorflow as tf

from eoflow.models.losses import TanimotoDistanceLoss
from eoflow.models.metrics import MCCMetric
from eoflow.models.segmentation_base import segmentation_metrics


def load_model_from_checkpoints(model_dir, model_name, model_f, cfg_name='model_cfg.json', build_shape=None):
    """ Load a model given a config json file """
    
    with open(f'{model_dir}/{model_name}/{cfg_name}', 'r') as jfile:
        model_cfg = json.load(jfile) 

    model = model_f(model_cfg)

    if build_shape is not None:
        model.build(build_shape)
        
    mcc_metric = MCCMetric(default_n_classes=model_cfg['n_classes'], default_threshold=.5)
    mcc_metric.init_from_config({'n_classes': model_cfg['n_classes']})

    model.net.compile(loss={'extent': TanimotoDistanceLoss(),
                            'boundary': TanimotoDistanceLoss(),
                            'distance': TanimotoDistanceLoss()},
                      metrics=[segmentation_metrics['accuracy'](),
                               tf.keras.metrics.MeanIoU(num_classes=model_cfg['n_classes']),
                               mcc_metric])
    
    model.net.load_weights(f'{model_dir}/{model_name}/checkpoints/model.ckpt')
    
    return model


def get_saliency_map(inputs, labels, model):
    """ Compute saliency map https://arxiv.org/pdf/1312.6034.pdf """
    with tf.GradientTape(persistent=True) as g_tape:
        g_tape.watch(inputs)
        loss = tf.losses.CategoricalCrossentropy()(tf.one_hot(labels, depth=2), model(inputs))    

    gradients_wrt_image = g_tape.gradient(loss, inputs, unconnected_gradients=tf.UnconnectedGradients.NONE)
    
    # as for original paper https://arxiv.org/pdf/1312.6034.pdf, Section 3.1
    saliency_map = np.max(np.abs(gradients_wrt_image.numpy()), axis=-1).squeeze()
    
    saliency_map = (saliency_map-saliency_map.min())/(saliency_map.max()-saliency_map.min())
    
    return saliency_map


def get_grad_cam_map(inputs, model, layer_name, class_idx, output_feature=0, eps=1e-16):
    """ Compute Grad-CAM map based on PyImageSerach implementation. Paper https://arxiv.org/pdf/1610.02391.pdf """
    
    grad_model = tf.keras.models.Model(inputs=[model.inputs],
                                       outputs=[model.get_layer(layer_name).output, model.output])
               
    # record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        (conv_outputs, predictions) = grad_model(inputs)
        loss = predictions[output_feature][..., class_idx]
        
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, conv_outputs)

    # compute the guided gradients
    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads
    
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guided_grads[0], axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    
    heatmap = (cam.numpy() - np.min(cam.numpy())) / (np.max(cam.numpy())-np.min(cam.numpy())+eps)
    
    return heatmap, loss.numpy()
