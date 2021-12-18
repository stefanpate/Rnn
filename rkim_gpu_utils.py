#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: 07-20-2018
# Email: robert.f.kim@gmail.com
# Description: GPU utilization functions for TF and keras w/ TF

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction, dev_num):
    """
    Preallocating GPU memory on a GPU device

    gpu_fraction: any number between 0 and 1 (ex. 0.5 = use 50% of available VRAM)
    dev_num: GPU device number in string (ex. '0,1' for using GPU #0 and #1)
    """

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=dev_num

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def run_keras():
    """
    Example function for pre-allocating for keras w/ TF backend
    The following will preallocate 90% of VRAM on three gpu cards (0, 1, and 2)
    """
    KTF.set_session(get_session(0.9, '0,1,2'))

    # Run rest of your code here ...

def run_tf():
    """
    Example function for pre-allocating for TF
    The following will preallocates 90% of VRAM on three gpu cards (0, 1, and 2)
    """
    sess = get_session(0.9, '0,1,2')
    with sess.as_default():
        # Run rest of your code here ...

