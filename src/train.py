#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from unet import DenoiseUNetModel
from batch_generator import BatchGenerator
from config import Configuration


def train():
    """
    Train phase main process
    :return:
    """
    config = Configuration()
    batch_generator = BatchGenerator(config=config)

    # create two list to store cost values
    train_loss = np.zeros(shape=(config.max_epoch, ))
    val_loss = np.zeros(shape=(config.max_epoch, ))

    # create folders
    if not os.path.exists(config.train_summary_root_dir):
        os.makedirs(config.train_summary_root_dir)

    if not os.path.exists(config.dump_model_para_root_dir):
        os.makedirs(config.dump_model_para_root_dir)

    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    with tf.compat.v1.Session(config=session_config) as sess:
        model = DenoiseUNetModel(config=config)
        print('\n\nModel initialized successfully.')

        train_writer = tf.compat.v1.summary.FileWriter(config.train_summary_root_dir, sess.graph)
        tf.compat.v1.global_variables_initializer().run()
        saver = tf.compat.v1.train.Saver(max_to_keep=None)

        print('Start to train model:')
        train_step = 0
        for e in range(config.max_epoch):
            print('\n=====================================  Epoch {} ====================================='.format(e+1))
            # training phase
            batch_generator.reset_training_batches()
            for batch in range(batch_generator.train_batch_amount):
                input_batch, gt_batch = batch_generator.next_train_batch()

                train_batch_loss, optimizer, summary_op, denoised_seq, total_params = sess.run(
                    fetches=[
                        model.loss,
                        model.optimizer,
                        model.summary_op,
                        model.denoised_seq,
                        model.all_trainable_vars
                    ],
                    feed_dict={
                        model.input_data: input_batch,
                        model.ground_truth: gt_batch
                    })

                # add summary and accumulate stats
                train_writer.add_summary(summary_op, train_step)
                train_loss[e] += train_batch_loss
                train_step += 1

                print('[Training] Epoch {}: batch {} / {}: loss = {}, # of parameters = {}.'.format(
                    e+1, batch, batch_generator.train_batch_amount, round(train_batch_loss, 4), total_params))

                # if (batch + 1) % 50 == 0:
                #     import matplotlib.pyplot as plt
                #     plt.figure()
                #     plt.subplot(311)
                #     plt.plot(input_batch[0])
                #     plt.title('noisy sequence')
                #     plt.subplot(312)
                #     plt.plot(gt_batch[0])
                #     plt.title('clean  sequence')
                #     plt.subplot(313)
                #     plt.plot(denoised_seq[0])
                #     plt.title('inferred sequence')
                #     plt.show()

            train_loss[e] /= batch_generator.train_batch_amount
            print('--------------------------------------------------------------------------------------------------')

            # validation phase
            for batch in range(batch_generator.val_batch_amount):
                input_batch, gt_batch = batch_generator.next_val_batch()

                val_batch_loss, denoised_seq = sess.run(
                    fetches=[
                        model.loss,
                        model.denoised_seq,
                    ],
                    feed_dict={
                        model.input_data: input_batch,
                        model.ground_truth: gt_batch
                    })

                val_loss[e] += val_batch_loss

                print('[Inference] Epoch {}: batch {}: loss = {}.'.format(
                    e+1,
                    batch,
                    round(val_batch_loss, 4)
                ))

            val_loss[e] /= batch_generator.val_batch_amount

            # checkpoint model variable
            if (e + 1) % config.save_every_epoch == 0:
                model_name = 'epoch_{}_train_loss_{:3f}_val_loss_{:3f}.ckpt'.format(
                    e + 1,
                    np.round(train_loss[e], 4),
                    np.round(val_loss[e], 4))
                dump_model_full_path = os.path.join(config.dump_model_para_root_dir, model_name)
                saver.save(sess=sess, save_path=dump_model_full_path)

        # close writer and session objects
        train_writer.close()
        sess.close()


if __name__ == '__main__':
    train()
