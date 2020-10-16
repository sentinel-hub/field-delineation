import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib as mpl
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
    def plot_predictions(input_image, lbl_extent, lbl_boundary, lbl_dist, pred_extent, pred_bound, pred_dist, n_classes):
        fig, ax = plt.subplots(4, 3, figsize=(25, 25))

        scaled_image = np.clip(input_image*2.5, 0., 1.)
        
        ax[0][0].imshow(scaled_image)
        ax[0][0].title.set_text('Input image')
        ax[1][0].imshow(scaled_image)
        ax[1][0].title.set_text('Input image')
        ax[2][0].imshow(scaled_image)
        ax[2][0].title.set_text('Input image')
        ax[3][0].imshow(scaled_image)
        ax[3][0].title.set_text('Input image')


        
        cnorm = mpl.colors.NoNorm()
        cmap = plt.cm.get_cmap('Set3', n_classes)
        # Extent 
        
        ax[0][1].imshow(lbl_extent, cmap=cmap, norm=cnorm)
        ax[0][1].title.set_text('Labels extent')

        img = ax[0][2].imshow(pred_extent, cmap=cmap, norm=cnorm)
        ax[0][2].title.set_text('Predictions extent')
        
        # Boundary 
        
        ax[1][1].imshow(lbl_boundary, cmap=cmap, norm=cnorm)
        ax[1][1].title.set_text('Labels boundary')

        img = ax[1][2].imshow(pred_bound, cmap=cmap, norm=cnorm)
        ax[1][2].title.set_text('Predictions boundary')

        # Distance 1 
          
        
        ax[2][1].imshow(lbl_dist.numpy()[..., 0].squeeze(), cmap='BuGn', vmin=0, vmax=1)
        ax[2][1].title.set_text('Labels distance (d)')
        
        
        img = ax[2][2].imshow(pred_dist[..., 0].squeeze(), cmap='Greens', vmin=0, vmax=1)
        ax[2][2].title.set_text('Predictions distance (p) ')
        
        
        ax[3][1].imshow(lbl_dist.numpy()[..., 1].squeeze(), cmap='BuGn', vmin=0, vmax=1)
        ax[3][1].title.set_text('Labels distance (1-d)')
        
        
        img = ax[3][2].imshow(pred_dist[..., 1].squeeze(), cmap='BuGn', vmin=0, vmax=1)
        ax[3][2].title.set_text('Predictions distance (1-p)')


    
        plt.colorbar(img, ax=[ax[0][0], ax[0][1], ax[0][2]], shrink=0.8, ticks=list(range(n_classes)))
        plt.colorbar(img, ax=[ax[1][0], ax[1][1], ax[1][2]], shrink=0.8, ticks=list(range(n_classes)))
        #plt.colorbar(img, ax=[ax[2][0], ax[2][1], ax[2][2]], shrink=0.8, ticks=list(range(n_classes)))


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
        
        viz_iter = zip(images, labels_extent, labels_boundary, labels_distance, preds_raw_extent, preds_raw_bound, preds_raw_dist)
        for image, lbl_extent, lbl_boundary, lbl_dist, pred_extent, pred_boundary, pred_distance in viz_iter:
            # Plot predictions and convert to image
            fig = self.plot_predictions(image, lbl_extent, lbl_boundary, lbl_dist, pred_extent, pred_boundary, pred_distance,  num_classes)
            img = plot_to_image(fig)
            vis_images.append(img)

        n_images = len(vis_images)
        vis_images = tf.concat(vis_images, axis=0)

        with self.file_writer.as_default():
            tf.summary.image('predictions', vis_images, step=step, max_outputs=n_images)

    def on_epoch_end(self, epoch, logs=None):
        self.prediction_summaries(epoch)

