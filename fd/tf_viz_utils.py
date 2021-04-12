#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import tensorflow as tf
from sklearn.cluster import KMeans

from eoflow.utils.tf_utils import plot_to_image


class ExtentBoundDistVisualizationCallback(tf.keras.callbacks.Callback):
    """ Keras Callback for saving prediction visualizations to TensorBoard. """

    def __init__(self, val_images, log_dir, time_index=0, rgb_indices=[2, 1, 0]):
        """
        :param val_images: Images to run predictions on. Tuple of (images, labels).
        :type val_images: (np.array, np.array)
        :param log_dir: Directory where the TensorBoard logs are written.
        :type log_dir: str
        :param time_index: Time index to use, when multiple time slices are available, defaults to 0
        :type time_index: int, optional
        :param rgb_indices: Indices for R, G and B bands in the input image, defaults to [0,1,2]
        :type rgb_indices: list, optional
        """
        super().__init__()

        self.val_images = val_images
        self.time_index = time_index
        self.rgb_indices = rgb_indices

        self.file_writer = tf.summary.create_file_writer(log_dir)

    @staticmethod
    def plot_predictions(input_image,
                         lbl_extent, lbl_boundary, lbl_dist,
                         pred_extent, pred_bound, pred_dist,
                         n_classes, nrows=4, ncols=3):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))

        scaled_image = np.clip(input_image*2.5, 0., 1.)

        for nrow in range(nrows):
            ax[nrow][0].imshow(scaled_image)
            ax[nrow][0].title.set_text('Input image')

        cnorm = mpl.colors.NoNorm()
        cmap = plt.cm.get_cmap('Set3', n_classes)

        # Extent
        ax[0][1].imshow(lbl_extent, cmap=cmap, norm=cnorm)
        ax[0][1].title.set_text('Labels extent')

        ax[0][2].imshow(pred_extent, cmap=cmap, norm=cnorm)
        ax[0][2].title.set_text('Predictions extent')
        
        # Boundary
        ax[1][1].imshow(lbl_boundary, cmap=cmap, norm=cnorm)
        ax[1][1].title.set_text('Labels boundary')

        ax[1][2].imshow(pred_bound, cmap=cmap, norm=cnorm)
        ax[1][2].title.set_text('Predictions boundary')

        # Distance 1
        ax[2][1].imshow(lbl_dist.numpy()[..., 0].squeeze(), cmap='BuGn', vmin=0, vmax=1)
        ax[2][1].title.set_text('Labels distance (d)')

        ax[2][2].imshow(pred_dist[..., 0].squeeze(), cmap='Greens', vmin=0, vmax=1)
        ax[2][2].title.set_text('Predictions distance (p) ')

        ax[3][1].imshow(lbl_dist.numpy()[..., 1].squeeze(), cmap='BuGn', vmin=0, vmax=1)
        ax[3][1].title.set_text('Labels distance (1-d)')
        img = ax[3][2].imshow(pred_dist[..., 1].squeeze(), cmap='BuGn', vmin=0, vmax=1)
        ax[3][2].title.set_text('Predictions distance (1-p)')

        plt.colorbar(img, ax=[ax[0][0], ax[0][1], ax[0][2]], shrink=0.8, ticks=list(range(n_classes)))
        plt.colorbar(img, ax=[ax[1][0], ax[1][1], ax[1][2]], shrink=0.8, ticks=list(range(n_classes)))

        return fig

    def prediction_summaries(self, step):
        images, labels = self.val_images

        labels_extent = labels['extent']
        labels_boundary = labels['boundary']
        labels_distance = labels['distance']
        
        images = images['features']
        preds_raw = self.model.predict(images)
        pred_shape = tf.shape(preds_raw)
        
        # Crop images and labels to output size
        labels_extent = tf.image.resize_with_crop_or_pad(labels_extent, pred_shape[2], pred_shape[3])
        labels_boundary = tf.image.resize_with_crop_or_pad(labels_boundary, pred_shape[2], pred_shape[3])
        labels_distance = tf.image.resize_with_crop_or_pad(labels_distance, pred_shape[2], pred_shape[3])

        images = tf.image.resize_with_crop_or_pad(images, pred_shape[2], pred_shape[3])
        # Take RGB values
        images = images.numpy()[..., self.rgb_indices]

        num_classes = labels_extent.shape[-1]

        # Get class ids
        preds_raw_extent = np.argmax(preds_raw[0], axis=-1)
        preds_raw_bound = np.argmax(preds_raw[1], axis=-1)
        preds_raw_dist = preds_raw[2]

        labels_extent = np.argmax(labels_extent, axis=-1)
        labels_boundary = np.argmax(labels_boundary, axis=-1)

        vis_images = []
        viz_iter = zip(images, labels_extent, labels_boundary, labels_distance,
                       preds_raw_extent, preds_raw_bound, preds_raw_dist)
        for image, lbl_extent, lbl_boundary, lbl_dist, pred_extent, pred_boundary, pred_distance in viz_iter:
            # Plot predictions and convert to image
            fig = self.plot_predictions(image,
                                        lbl_extent, lbl_boundary, lbl_dist,
                                        pred_extent, pred_boundary, pred_distance,  num_classes)
            img = plot_to_image(fig)
            vis_images.append(img)

        n_images = len(vis_images)
        vis_images = tf.concat(vis_images, axis=0)

        with self.file_writer.as_default():
            tf.summary.image('predictions', vis_images, step=step, max_outputs=n_images)

    def on_epoch_end(self, epoch, logs=None):
        self.prediction_summaries(epoch)


def plot_layer_activations(model, batch_features, layer_names=None, order=False, show_centroids=False, basename=None):
    """ Util to plot activations of all conv/pyramid layers of the model. Images are saved if basename is provided. """
    if not layer_names:
        layer_names = [layer.name for layer in model.layers if any([c in layer.name for c in ['conv2d', 'pyramid']])]

    for layer_name in layer_names:
        intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                                         outputs=model.get_layer(layer_name).output)

        intermediate_output = intermediate_layer_model.predict(batch_features)

        plot_activations(intermediate_output.squeeze(), layer_name, order, show_centroids, basename=basename)


def plot_activations(activations, layer, order=False, show_centroids=False, basename=None):
    height, width, nfeats = activations.shape
    title = f'Activations for layer {layer} - {nfeats} features'
    if order and nfeats > 8:
        kmeans = KMeans(n_clusters=nfeats // 8, random_state=42)
        clusters = kmeans.fit_predict(activations.reshape(height * width, nfeats).T)
        activations = activations[..., np.argsort(clusters)]
        title = f'Activations for layer {layer} - {nfeats} features - ordered'
    if show_centroids and nfeats > 8:
        kmeans = KMeans(n_clusters=nfeats // 8, random_state=42)
        kmeans.fit(activations.reshape(height * width, nfeats).T)
        activations = kmeans.cluster_centers_.T.reshape(height, width, nfeats // 8)
        title = f'Activations for layer {layer} - {nfeats} features - centroids only'
        nfeats = nfeats // 8
    ncols = 8 if nfeats % 8 == 0 else nfeats
    nrows = nfeats // ncols if nfeats % 8 == 0 else 1
    fig, ax = plt.subplots(ncols=ncols,
                           nrows=nrows,
                           sharex='all',
                           sharey='all',
                           figsize=(15, 15 * (nrows / ncols)))
    for nfeat in range(nfeats):
        if len(ax.shape) == 2:
            ax[nfeat // ncols][nfeat % ncols].imshow(activations[..., nfeat], cmap='gray')
        else:
            ax[nfeat % ncols].imshow(activations[..., nfeat], cmap='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title)
    if basename:
        plt.savefig(f'{basename}_layer-{layer}_nfeats-{nfeats}.png')


def plot_input_data(input_data):
    n_bands = input_data.shape[-1]
    fig, ax = plt.subplots(ncols=5, nrows=1, sharey='all', figsize=(15, 5))
    ax[0].imshow(input_data[..., [0, 1, 2]])
    for ib in range(n_bands):
        ax[ib + 1].imshow(input_data[..., ib], vmin=0, vmax=1, cmap='gray')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Input image')


def plot_conv_activations(model, input_data, order=False, show_centroids=False):
    """ Util to plot activations of convolutional layers for trained model given input_data """
    plot_input_data(input_data.squeeze())

    layer_names = [layer.name for layer in model.layers[0].layers]
    conv_layers = [layer_name for layer_name in layer_names if layer_name.startswith('conv2d')]
    softmax = [layer_name for layer_name in layer_names if layer_name.startswith('softmax')]

    for layer_name in conv_layers + softmax:
        intermediate_layer_model = tf.keras.models.Model(inputs=model.layers[0].input,
                                                         outputs=model.layers[0].get_layer(layer_name).output)

        intermediate_output = intermediate_layer_model.predict(input_data)

        plot_activations(intermediate_output.squeeze(), layer_name, order, show_centroids)
