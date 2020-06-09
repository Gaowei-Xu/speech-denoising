#!/usr/bin/python3
# -*-coding:utf-8-*-
import os


class Configuration(object):
    def __init__(self):
        self._train_val_clean_wav = '../traindata/clean/'
        self._train_val_noisy_wav = '../traindata/noisy/'

        self._input_samples_dimen = 2048

        self._cache_root_dir = '../cache/'
        if not os.path.exists(self._cache_root_dir):
            os.makedirs(self._cache_root_dir)

        self._batch_size = 256

        self._learning_rate = 0.001

        self._max_epoch = 250
        self._train_summary_root_dir = '../train/'
        self._dump_model_para_root_dir = '../models/'
        self._save_every_epoch = 1

    @property
    def train_val_clean_wav(self):
        return self._train_val_clean_wav

    @property
    def train_val_noisy_wav(self):
        return self._train_val_noisy_wav

    @property
    def input_samples_dimen(self):
        return self._input_samples_dimen

    @property
    def cache_root_dir(self):
        return self._cache_root_dir

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def train_summary_root_dir(self):
        return self._train_summary_root_dir

    @property
    def dump_model_para_root_dir(self):
        return self._dump_model_para_root_dir

    @property
    def save_every_epoch(self):
        return self._save_every_epoch
