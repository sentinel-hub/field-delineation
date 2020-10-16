import tensorflow as tf

from eoflow.models import FCNModel
from eoflow.models.layers import Conv2D, Deconv2D, CropAndConcat, ResConv2D, PyramidPoolingModule


class ResUnetA(FCNModel):
    """
    ResUnetA

     https://github.com/feevos/resuneta/tree/145be5519ee4bec9a8cce9e887808b8df011f520/models

    """

    def build(self, inputs_shape):
        """Builds the net for input x."""
        x = tf.keras.layers.Input(shape=inputs_shape['features'][1:], name='features')
        dropout_rate = 1 - self.config.keep_prob

        # block 1
        initial_conv = Conv2D(
            filters=self.config.features_root,
            kernel_size=1,  # 1x1 kernel
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            dropout_rate=dropout_rate,
            use_bias=self.config.use_bias,
            batch_normalization=True,  # maybe
            padding=self.config.padding,
            num_repetitions=1)(x)

        # block 2
        resconv_1 = ResConv2D(
                filters=self.config.features_root,
                kernel_size=self.config.conv_size,
                dilation=[1, 3, 15, 31],
                strides=self.config.conv_stride,
                add_dropout=self.config.add_dropout,
                dropout_rate=dropout_rate,
                use_bias=self.config.use_bias,
                batch_normalization=self.config.add_batch_norm,
                padding=self.config.padding,
                num_parallel=4)(initial_conv)

        # block 3
        pool_1 = Conv2D(
            filters=2 * self.config.features_root,
            kernel_size=self.config.pool_size,
            strides=self.config.pool_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding='SAME',
            num_repetitions=1)(resconv_1)

        # block 4
        resconv_2 = ResConv2D(
            filters=2 * self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1, 3, 15, 31],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=4)(pool_1)

        # block 5
        pool_2 = Conv2D(
            filters=4 * self.config.features_root,
            kernel_size=self.config.pool_size,
            strides=self.config.pool_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding='SAME',
            num_repetitions=1)(resconv_2)

        # block 6
        resconv_3 = ResConv2D(
            filters=4 * self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1, 3, 15],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=3)(pool_2)

        # block 7
        pool_3 = Conv2D(
            filters=8 * self.config.features_root,
            kernel_size=self.config.pool_size,
            strides=self.config.pool_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding='SAME',
            num_repetitions=1)(resconv_3)

        # block 8
        resconv_4 = ResConv2D(
            filters=8 * self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1, 3, 15],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=3)(pool_3)

        # block 9
        pool_4 = Conv2D(
            filters=16 * self.config.features_root,
            kernel_size=self.config.pool_size,
            strides=self.config.pool_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding='SAME',
            num_repetitions=1)(resconv_4)

        # block 10
        resconv_5 = ResConv2D(
            filters=16 * self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=1)(pool_4)

        # block 11
        pool_5 = Conv2D(
            filters=32 * self.config.features_root,
            kernel_size=self.config.pool_size,
            strides=self.config.pool_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding='SAME',
            num_repetitions=1)(resconv_5)

        # block 12
        resconv_6 = ResConv2D(
            filters=32 * self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=1)(pool_5)

        # block 13
        ppm1 = PyramidPoolingModule(filters=32 * self.config.features_root,
                                    batch_normalization=True)(resconv_6)

        # block 14
        deconv_1 = Deconv2D(
            filters=32 * self.config.features_root,
            kernel_size=self.config.deconv_size,
            batch_normalization=self.config.add_batch_norm)(ppm1)

        # block 15
        concat_1 = CropAndConcat()(resconv_5, deconv_1)
        concat_1 = Conv2D(
            filters=16 * self.config.features_root,
            kernel_size=1,  # 1x1 kernel
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=True,  # maybe
            padding=self.config.padding,
            num_repetitions=1)(concat_1)

        # block 16
        resconv_7 = ResConv2D(
            filters=16 * self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=1)(concat_1)

        # block 17
        deconv_2 = Deconv2D(
            filters=16 * self.config.features_root,
            kernel_size=self.config.deconv_size,
            batch_normalization=self.config.add_batch_norm)(resconv_7)

        # block 18
        concat_2 = CropAndConcat()(resconv_4, deconv_2)
        concat_2 = Conv2D(
            filters=8 * self.config.features_root,
            kernel_size=1,  # 1x1 kernel
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=True,  # maybe
            padding=self.config.padding,
            num_repetitions=1)(concat_2)

        # block 19
        resconv_8 = ResConv2D(
            filters=8 * self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1, 3, 15],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=3)(concat_2)

        # block 20
        deconv_3 = Deconv2D(
            filters=8 * self.config.features_root,
            kernel_size=self.config.deconv_size,
            batch_normalization=self.config.add_batch_norm)(resconv_8)

        # block 21
        concat_3 = CropAndConcat()(resconv_3, deconv_3)
        concat_3 = Conv2D(
            filters=4 * self.config.features_root,
            kernel_size=1,  # 1x1 kernel
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=True,  # maybe
            padding=self.config.padding,
            num_repetitions=1)(concat_3)

        # block 22
        resconv_9 = ResConv2D(
            filters=4 * self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1, 3, 15],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=3)(concat_3)

        # block 23
        deconv_4 = Deconv2D(
            filters=4 * self.config.features_root,
            kernel_size=self.config.deconv_size,
            batch_normalization=self.config.add_batch_norm)(resconv_9)

        # block 24
        concat_4 = CropAndConcat()(resconv_2, deconv_4)
        concat_4 = Conv2D(
            filters=2 * self.config.features_root,
            kernel_size=1,  # 1x1 kernel
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=True,  # maybe
            padding=self.config.padding,
            num_repetitions=1)(concat_4)

        # block 25
        resconv_10 = ResConv2D(
            filters=2 * self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1, 3, 15, 31],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=4)(concat_4)

        # block 26
        deconv_5 = Deconv2D(
            filters=2 * self.config.features_root,
            kernel_size=self.config.deconv_size,
            batch_normalization=self.config.add_batch_norm)(resconv_10)

        # block 27
        concat_5 = CropAndConcat()(resconv_1, deconv_5)
        concat_5 = Conv2D(
            filters=self.config.features_root,
            kernel_size=1,  # 1x1 kernel
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=True,  # maybe
            padding=self.config.padding,
            num_repetitions=1)(concat_5)

        # block 28
        resconv_11 = ResConv2D(
            filters=self.config.features_root,
            kernel_size=self.config.conv_size,
            dilation=[1, 3, 15, 31],
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            padding=self.config.padding,
            num_parallel=4)(concat_5)

        # block 29
        concat_6 = CropAndConcat()(initial_conv, resconv_11)
        concat_6 = Conv2D(
            filters=self.config.features_root,
            kernel_size=1,  # 1x1 kernel
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            use_bias=self.config.use_bias,
            dropout_rate=dropout_rate,
            batch_normalization=True,  # maybe
            padding=self.config.padding,
            num_repetitions=1)(concat_6)

        # block 30
        ppm2 = PyramidPoolingModule(filters=self.config.features_root,
                                    batch_normalization=True)(concat_6)

        # comditioned multi-tasking
        # first get distance
        distance_conv = Conv2D(
            filters=self.config.features_root,
            kernel_size=self.config.conv_size,
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            num_repetitions=2,
            padding=self.config.padding)(concat_6)  # in last layer we take the combined features
        logits_distance = tf.keras.layers.Conv2D(filters=self.config.n_classes, kernel_size=1)(distance_conv)
        logits_distance = tf.keras.layers.Softmax(name='distance')(logits_distance)

        # concatenate distance logits to features
        dcc = CropAndConcat()(ppm2, logits_distance)
        boundary_conv = Conv2D(
            filters=self.config.features_root,
            kernel_size=self.config.conv_size,
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            num_repetitions=1,
            padding=self.config.padding)(dcc)
        logits_boundary = tf.keras.layers.Conv2D(filters=self.config.n_classes, kernel_size=1)(boundary_conv)
        logits_boundary = tf.keras.layers.Softmax(name='boundary')(logits_boundary)

        bdcc = CropAndConcat()(dcc, logits_boundary)
        extent_conv = Conv2D(
            filters=self.config.features_root,
            kernel_size=self.config.conv_size,
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            num_repetitions=2,
            padding=self.config.padding)(bdcc)
        logits_extent = tf.keras.layers.Conv2D(filters=self.config.n_classes, kernel_size=1)(extent_conv)
        logits_extent = tf.keras.layers.Softmax(name='extent')(logits_extent)

        self.net = tf.keras.Model(inputs=x, outputs=[logits_extent, logits_boundary, logits_distance])

    def call(self, inputs, training=True):
        return self.net(inputs, training)
