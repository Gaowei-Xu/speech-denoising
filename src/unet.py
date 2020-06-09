#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


class DenoiseUNetModel(object):
    def __init__(self, config):
        """
        configure model parameters and build model
        :param config: model configuration parameters
        """
        self._config = config

        self._input_data = tf.compat.v1.placeholder(
            tf.float32,
            shape=[
                self._config.batch_size,
                self._config.input_samples_dimen],
            name="input_noisy_seq")

        self._ground_truth = tf.compat.v1.placeholder(
            tf.float32,
            shape=[
                self._config.batch_size,
                self._config.input_samples_dimen],
            name="gt_clean_seq")

        self._global_step = tf.Variable(0, trainable=False)
        self._optimizer, self._summary_op = None, None
        self._all_trainable_vars = None
        self._denoised_seq = None
        self._loss = None

        self.build_model()

    def build_model(self):
        """
        build U-Net model
        :return:
        """
        def conv1d_transpose(x, filters, kernel_size, padding, output_width, strides, name):
            weights = tf.Variable(
                initial_value=tf.random.uniform(
                    shape=[kernel_size, filters, x.shape[-1]],
                    minval=0.0,
                    maxval=0.02
                ),
                trainable=True,
                name='conv1d_transpose_weights_{}'.format(name))

            output = tf.nn.conv1d_transpose(
                x,
                filters=weights,                                # [filter_width, output_channels, in_channels]
                output_shape=[self._config.batch_size, output_width, filters],   # [batch size, width, channels]
                strides=strides,
                padding=padding)
            return tf.nn.leaky_relu(output)

        with tf.compat.v1.variable_scope("U-Net-1D-BackBone"):
            # net down
            input_data = tf.reshape(self._input_data,
                                    shape=[
                                        self._config.batch_size,            # batch size
                                        self._config.input_samples_dimen,   # width
                                        1])                                 # channels
            # filters shape: [filter_width, in_channels, out_channels]
            conv_1 = tf.keras.layers.Conv1D(filters=24, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(input_data)
            conv_2 = tf.keras.layers.Conv1D(filters=24, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_1)
            pool_1 = tf.nn.max_pool1d(conv_2, ksize=2, strides=2, padding='SAME')

            conv_3 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(pool_1)
            conv_4 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_3)
            pool_2 = tf.nn.max_pool1d(conv_4, ksize=2, strides=2, padding='SAME')

            conv_5 = tf.keras.layers.Conv1D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(pool_2)
            conv_6 = tf.keras.layers.Conv1D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_5)
            pool_3 = tf.nn.max_pool1d(conv_6, ksize=2, strides=2, padding='SAME')

            conv_7 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(pool_3)
            conv_8 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_7)
            pool_4 = tf.nn.max_pool1d(conv_8, ksize=2, strides=2, padding='SAME')

            # bottom
            conv_9 = tf.keras.layers.Conv1D(filters=96, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(pool_4)
            conv_10 = tf.keras.layers.Conv1D(filters=96, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_9)

            # net up
            upconv_1 = conv1d_transpose(x=conv_10, filters=96, kernel_size=3, padding='SAME', output_width=conv_8.shape[1], strides=2, name='upconv_1')
            concat_1 = tf.concat([upconv_1, conv_8], axis=-1)
            conv_11 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(concat_1)
            conv_12 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_11)

            upconv_2 = conv1d_transpose(x=conv_12, filters=64, kernel_size=3, padding='SAME', output_width=conv_6.shape[1], strides=2, name='upconv_2')
            concat_2 = tf.concat([upconv_2, conv_6], axis=-1)
            conv_13 = tf.keras.layers.Conv1D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(concat_2)
            conv_14 = tf.keras.layers.Conv1D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_13)

            upconv_3 = conv1d_transpose(x=conv_14, filters=48, kernel_size=3, padding='SAME', output_width=conv_4.shape[1], strides=2, name='upconv_3')
            concat_3 = tf.concat([upconv_3, conv_4], axis=-1)
            conv_15 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(concat_3)
            conv_16 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_15)

            upconv_4 = conv1d_transpose(x=conv_16, filters=32, kernel_size=3, padding='SAME', output_width=conv_2.shape[1], strides=2, name='upconv_4')
            concat_4 = tf.concat([upconv_4, conv_2], axis=-1)
            conv_17 = tf.keras.layers.Conv1D(filters=24, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(concat_4)
            conv_18 = tf.keras.layers.Conv1D(filters=24, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_17)
            conv_19 = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu)(conv_18)
            self._denoised_seq = tf.reshape(conv_19, shape=[self._config.batch_size, self._config.input_samples_dimen])

        with tf.compat.v1.variable_scope("Loss"):
            flatten_ground_truth = tf.reshape(self._ground_truth, [self._config.batch_size, -1])
            flatten_denoised_seq = tf.reshape(conv_19, [self._config.batch_size, -1])

            flatten_loss = tf.losses.mean_absolute_percentage_error(
                y_true=flatten_ground_truth,
                y_pred=flatten_denoised_seq
            )
            self._loss = tf.reduce_mean(flatten_loss)
            tf.compat.v1.summary.scalar('loss', self._loss)

        with tf.compat.v1.variable_scope("params-stat"):
            self._all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.compat.v1.trainable_variables()])

        with tf.compat.v1.variable_scope("optimization"):
            train_op = tf.compat.v1.train.AdamOptimizer(self._config.learning_rate)
            self._optimizer = train_op.minimize(self._loss)
            self._summary_op = tf.compat.v1.summary.merge_all()

    @property
    def loss(self):
        return self._loss

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def ground_truth(self):
        return self._ground_truth

    @property
    def input_data(self):
        return self._input_data

    @property
    def denoised_seq(self):
        return self._denoised_seq

    @property
    def all_trainable_vars(self):
        return self._all_trainable_vars

