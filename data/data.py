#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:00:55 2018

@author: siqihao
"""

import numpy as np
import pickle as pkl


def sample_params(size):
    pitch = np.array([np.random.uniform(20, 20000) for _ in range(size)])
    sampling_freq = np.array([np.random.uniform(5, 10) * 10000 for i in range(size)])
    stretch_factor = np.array([np.random.uniform(1, 10) for _ in range(size)])
    flag = np.array([np.random.uniform(0, 1) for _ in range(size)])
    cqt_spec = np.array([np.zeros(784).reshape(28, 28) for _ in range(size)])
    return pitch, sampling_freq, stretch_factor, flag, cqt_spec
        

def generate_data(file, size):
    with open(file, 'wb') as fh:
        pitch, sampling_freq, stretch_factor, flag, cqt_spec = sample_params(size)
        data_dict = {'parameters' : np.array([pitch, sampling_freq, stretch_factor, flag]), 'cqt_spec' : cqt_spec}
        fh.write(pkl.dumps(data_dict))
    fh.close()
    print(file)
   
    
def read_data(file):
    with open(file, 'rb') as fh:
        data = pkl.loads(fh.read())
    fh.close()
    return data


def create_datasets():
    generate_data('eval.pkl', 100)
    generate_data('test.pkl', 5000)
    generate_data('train.pkl', 50000)

    
def read_dataset():
    return read_data('train.pkl'), read_data('test.pkl'), read_data('eval.pkl')