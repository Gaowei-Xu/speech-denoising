#!/usr/bin/python3
# -*-coding:utf-8-*-
import os
import numpy as np
import pickle
import wave
import random
from config import Configuration


class BatchGenerator(object):
    def __init__(self, config):
        self._config = config

        # collect all samples
        # self.collect_samples()

        self._train_batch_full_path = [
            os.path.join(self._config.cache_root_dir, f) for f in
            os.listdir(self._config.cache_root_dir) if f.startswith('train_batch_')]

        self._val_batch_full_path = [
            os.path.join(self._config.cache_root_dir, f) for f in
            os.listdir(self._config.cache_root_dir) if f.startswith('val_batch_')]

        self._train_batch_amount = len(self._train_batch_full_path)
        self._val_batch_amount = len(self._val_batch_full_path)

        self._train_batch_index, self._val_batch_index = 0, 0

    def collect_samples(self, train_val_ratio=5.0):
        samples = list()

        noisy_wav_files = [f for f in os.listdir(self._config.train_val_noisy_wav) if f.endswith('.wav')]

        for index, wav_name in enumerate(noisy_wav_files):
            print('Collecting samples from wav file {} {}/{}...'.format(wav_name, index+1, len(noisy_wav_files)))
            noisy_wav_full_path = os.path.join(self._config.train_val_noisy_wav, wav_name)
            clean_wav_full_path = os.path.join(self._config.train_val_clean_wav, wav_name)
            noisy_rf = wave.open(noisy_wav_full_path, 'rb')
            clean_rf = wave.open(clean_wav_full_path, 'rb')
            (noisy_channels, noisy_sampling_with, noisy_sampling_freq,
             noisy_sampling_points, noisy_comp_type, noisy_comp_desc) = noisy_rf.getparams()
            (clean_channels, clean_sampling_with, clean_sampling_freq,
             clean_sampling_points, clean_comp_type, clean_comp_desc) = clean_rf.getparams()

            assert noisy_channels == clean_channels
            assert noisy_sampling_with == 2 and noisy_channels == 1
            assert noisy_sampling_with == clean_sampling_with
            assert noisy_sampling_freq == clean_sampling_freq
            assert noisy_sampling_points == clean_sampling_points
            assert noisy_comp_type == clean_comp_type
            assert noisy_comp_desc == clean_comp_desc

            noisy_seq_str = noisy_rf.readframes(noisy_sampling_points)
            clean_seq_str = clean_rf.readframes(clean_sampling_points)
            noisy_seq = np.frombuffer(noisy_seq_str, dtype=np.short)
            clean_seq = np.frombuffer(clean_seq_str, dtype=np.short)

            # TODO: could apply better normalization, here use 16bit (1bit for sign) to normalize sequence
            noisy_seq = (noisy_seq + 32768.0) / 65536.0
            clean_seq = (clean_seq + 32768.0) / 65536.0

            segments = len(noisy_seq) // self._config.input_samples_dimen
            for k in range(segments):
                samples.append(dict(
                    noisy_seq=noisy_seq[
                              k*self._config.input_samples_dimen:
                              (k+1)*self._config.input_samples_dimen],
                    clean_seq=clean_seq[
                              k * self._config.input_samples_dimen:
                              (k + 1) * self._config.input_samples_dimen]
                ))

            if segments * self._config.input_samples_dimen != len(noisy_seq):
                noisy_sample_with_zero_padding = np.zeros(shape=(self._config.input_samples_dimen, ))
                noisy_sample_with_zero_padding[0: len(noisy_seq) - segments * self._config.input_samples_dimen] = \
                    noisy_seq[segments * self._config.input_samples_dimen:]

                clean_sample_with_zero_padding = np.zeros(shape=(self._config.input_samples_dimen, ))
                clean_sample_with_zero_padding[0: len(clean_seq) - segments * self._config.input_samples_dimen] = \
                    clean_seq[segments * self._config.input_samples_dimen:]

                samples.append(dict(
                    noisy_seq=noisy_sample_with_zero_padding,
                    clean_seq=clean_sample_with_zero_padding
                ))

        random.shuffle(samples)
        batch_amount = len(samples) // self._config.batch_size

        train_index, val_index = 0, 0
        for batch_index in range(batch_amount):
            if batch_index < int(batch_amount * train_val_ratio / (1.0 + train_val_ratio)):
                # train batch
                train_sample = dict(
                    input_batch=np.zeros(shape=(self._config.batch_size, self._config.input_samples_dimen)),
                    gt_batch=np.zeros(shape=(self._config.batch_size, self._config.input_samples_dimen))
                )

                for q, k in enumerate(
                        range(batch_index * self._config.batch_size,
                              (batch_index + 1) * self._config.batch_size)):
                    train_sample['input_batch'][q] = samples[k]['noisy_seq']
                    train_sample['gt_batch'][q] = samples[k]['clean_seq']

                train_index += 1
                with open(os.path.join(self._config.cache_root_dir, 'train_batch_{}.pkl'.format(train_index)), 'wb') as wf:
                    pickle.dump(train_sample, wf)
            else:
                # validation batch
                val_sample = dict(
                    input_batch=np.zeros(shape=(self._config.batch_size, self._config.input_samples_dimen)),
                    gt_batch=np.zeros(shape=(self._config.batch_size, self._config.input_samples_dimen))
                )

                for q, k in enumerate(
                        range(batch_index * self._config.batch_size,
                              (batch_index + 1) * self._config.batch_size)):
                    val_sample['input_batch'][q] = samples[k]['noisy_seq']
                    val_sample['gt_batch'][q] = samples[k]['clean_seq']

                val_index += 1
                with open(os.path.join(self._config.cache_root_dir, 'val_batch_{}.pkl'.format(val_index)), 'wb') as wf:
                    pickle.dump(val_sample, wf)

    def next_train_batch(self):
        with open(os.path.join(
                self._config.cache_root_dir,
                self._train_batch_full_path[self._train_batch_index]), 'rb') as rf:
            blob = pickle.load(rf)
            input_batch, gt_batch = blob['input_batch'], blob['gt_batch']

        self._train_batch_index += 1
        return input_batch, gt_batch

    def next_val_batch(self):
        with open(os.path.join(
                self._config.cache_root_dir,
                self._val_batch_full_path[self._val_batch_index]), 'rb') as rf:
            blob = pickle.load(rf)
            input_batch, gt_batch = blob['input_batch'], blob['gt_batch']

        self._val_batch_index += 1
        return input_batch, gt_batch

    @property
    def train_batch_amount(self):
        return self._train_batch_amount

    @property
    def val_batch_amount(self):
        return self._val_batch_amount

    def reset_validation_batches(self):
        self._val_batch_index = 0

    def reset_training_batches(self):
        random.shuffle(self._train_batch_full_path)
        self._train_batch_index = 0


if __name__ == '__main__':
    batch_generator = BatchGenerator(
        config=Configuration()
    )

    input_batch, gt_batch = batch_generator.next_train_batch()
    print(input_batch.shape)
    print(np.max(input_batch))
    print(np.min(input_batch))
    print(gt_batch.shape)
    print(np.max(gt_batch))
    print(np.min(gt_batch))
