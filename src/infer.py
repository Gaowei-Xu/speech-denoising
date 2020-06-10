#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from unet import DenoiseUNetModel
from config import Configuration
import wave
from audio import Audio
import matplotlib.pyplot as plt


def load(audio_full_path):
    rf = wave.open(audio_full_path, 'rb')

    # 声道数, 量化位数（byte), 采样频率, 采样点数, 压缩类型, 压缩类型的描述
    (channels, sampling_with, sampling_freq, sampling_points, comp_type, comp_desc) = rf.getparams()

    # TODO: here we assume that sampling width is 2 bytes and audio channels equals to 1
    assert sampling_with == 2
    assert channels == 1
    assert sampling_freq == 16000

    audio_seq_str = rf.readframes(sampling_points)
    audio_seq = np.frombuffer(audio_seq_str, dtype=np.short)

    # assign loaded .wav object
    audio = Audio(
        channels=channels,
        sampling_with=sampling_with,
        sampling_freq=sampling_freq,
        sampling_points=sampling_points,
        comp_type=comp_type,
        comp_desc=comp_desc,
        audio_seq=audio_seq  # numerical representation of input audio sequence
    )
    rf.close()
    return audio


def dump(audio, dump_full_path):
    wf = wave.open(dump_full_path, 'wb')

    # set wav parameters
    wf.setnchannels(audio.channels)
    wf.setsampwidth(audio.sampling_with)
    wf.setframerate(audio.sampling_freq)

    # dump audio sequence after de-noising
    audio.audio_seq = audio.audio_seq.astype(np.short)
    wf.writeframes(audio.audio_seq.tostring())
    wf.close()


def extract_batches(audio, config):
    seq = audio.audio_seq
    seq = (seq + 32768.0) / 65536.0
    batch_amount = len(seq) // (config.input_samples_dimen * config.batch_size)
    batches = list()
    for i in range(batch_amount):
        blob = np.zeros(shape=(config.batch_size, config.input_samples_dimen))
        offset = i * config.batch_size * config.input_samples_dimen
        for j in range(config.batch_size):
            blob[j] = seq[offset + j * config.input_samples_dimen: offset + (j+1) * config.input_samples_dimen]
        batches.append(blob)

    return batches


def inference(input_audio_path, dump_full_path):
    audio = load(input_audio_path)
    config = Configuration()

    batches = extract_batches(audio, config)

    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    predicted_clean_seq = list()
    with tf.compat.v1.Session(config=session_config) as sess:
        model = DenoiseUNetModel(config=config)
        tf.compat.v1.global_variables_initializer().run()
        saver = tf.compat.v1.train.Saver(max_to_keep=None)
        saver.restore(sess,
                      os.path.join(
                          config.dump_model_para_root_dir,
                          config.selected_model_name)
                      )

        for index in range(len(batches)):
            input_batch = batches[index]
            denoised_seq = sess.run(
                fetches=[
                    model.denoised_seq,
                ],
                feed_dict={
                    model.input_data: input_batch,
                    model.ground_truth: input_batch
                })

            flatten = np.reshape(denoised_seq, newshape=(config.batch_size * config.input_samples_dimen, ))
            flatten = flatten * 65536.0 - 32768.0
            flatten = flatten.astype(np.short)
            predicted_clean_seq.extend(list(flatten))

        audio.audio_seq = np.array(predicted_clean_seq)
        dump(audio, dump_full_path)


if __name__ == '__main__':
    inference(
        input_audio_path='../test_data/对话场景_过滤前.wav',
        dump_full_path='../test_data/对话场景_1D_UNet_过滤后.wav'
    )

