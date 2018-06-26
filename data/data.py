#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:00:55 2018

@author: siqihao
"""

import numpy as np
import pickle as pkl
import karplus_strong
import cqt_transform


def pad_zeros(image, shape):
    result = np.zeros(shape)
    result[:image.shape[0],:image.shape[1]] = image
    return result


def sample_params(size):
    pitch = np.array([np.random.uniform(20, 2000) for _ in range(size)])
    sampling_freq = np.array([np.random.uniform(5, 10) * 1000 for i in range(size)])
    stretch_factor = np.array([np.random.uniform(1, 10) for _ in range(size)])
    flag = np.array([np.random.uniform(0, 1) for _ in range(size)])
    #  ipdb.set_trace()
    samples = []
    strings = []
    cqt_specs = []
    for i in range(size):
        strings.append(karplus_strong.karplus_strong(pitch[i], 2 * sampling_freq[i], stretch_factor[i], 1))
        samples.append(strings[i].get_samples())
        cqt_spec = cqt_transform.compute_cqt_spec(samples[i]).T
        cqt_specs.append(pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1])))
    cqt_specs = np.array(cqt_specs)
    print(cqt_specs.shape)
    return pitch, sampling_freq, stretch_factor, flag, cqt_specs
        

def generate_data(file, size):
    with open(file, 'wb') as fh:
        pitch, sampling_freq, stretch_factor, flag, cqt_specs = sample_params(size)
        data_dict = {'parameters' : np.array([pitch, sampling_freq, stretch_factor, flag]).T, 'cqt_spec' : cqt_specs}
#         fh.write(pkl.dumps(data_dict))
        pkl.dump(data_dict, fh)
    fh.close()
    print(file)
   
    
def read_data(file):
    with open(file, 'rb') as fh:
        data = pkl.loads(fh.read())
    fh.close()
    return data


def create_datasets():
    generate_data('eval.pkl', 20)
    generate_data('test.pkl', 50)
    generate_data('train.pkl', 50)

    
def read_dataset():
    return read_data('train.pkl'), read_data('test.pkl'), read_data('eval.pkl')