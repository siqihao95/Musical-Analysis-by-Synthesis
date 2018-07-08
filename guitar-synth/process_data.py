#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:24:48 2018

@author: siqihao
"""

import sys
import pdb
#import ipdb
import math
import h5py
import pickle as pkl

import librosa
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from guitar import Guitar
import sequencer
sys.path.insert(0, '../models')
import cqt_transform
import utils

device = 'cuda'

num_frets = 20


def cqt_specgram(audio, n_bins, bins_per_octave, hop_length, sr, fmin, filter_scale):
    '''
    :param audio:
    :param sr:
    :return: shape = (n_bins, t)
    '''
    c = librosa.cqt(audio, sr = sr, n_bins = n_bins, bins_per_octave = bins_per_octave, hop_length = hop_length,
                    fmin = fmin, filter_scale = filter_scale)
    mag, phase = librosa.core.magphase(c)
    c_p = librosa.amplitude_to_db(mag, amin=1e-13, top_db=120., ref=np.max) / 120.0 + 1.0
    return c_p

def compute_cqt_spec(audio, n_bins = 84 * 4, bins_per_octave=12 * 4, hop_length = 256, sr = 16000, fmin = librosa.note_to_hz('C1'),
             filter_scale = 0.8):
    return cqt_specgram(audio, n_bins, bins_per_octave, hop_length, sr, fmin, filter_scale)


class Options:
    
    def __init__(self, character_variation = 0.5, string_damping=0.5, string_damping_variation=0.25, pluck_damping=0.5,
                pluck_damping_variation=0.25, string_tension=0.1, stereo_spread=0.2, string_damping_calculation='magic', 
                body='simple', mode='karplus-strong'):

        self.character_variation = character_variation
        self.string_damping=string_damping
        self.string_damping_variation=string_damping_variation
        self.pluck_damping=pluck_damping
        self.pluck_damping_variation=pluck_damping_variation
        self.string_tension=string_tension
        self.stereo_spread=stereo_spread
        self.string_damping_calculation=string_damping_calculation
        self.body=body
        self.mode=mode
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(9 * 108 * 108, 360)
        self.fc2 = nn.Linear(360, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 9 * 108 * 108)  # -1 is the batch_size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)    
        x = self.fc3(x)
        return x
    

class Net_pitch_sf(nn.Module):
    def __init__(self):
        super(Net_pitch_sf, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(9 * 108 * 108, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.pool(F.relu(self.conv1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 9 * 108 * 108)  # -1 is the batch_size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x1 = F.sigmoid(0.5 * x[:, np.arange(6)])
        x2 = torch.cat([x1, x[:, 6].unsqueeze(1)], dim=1)
        x = torch.cat([x2, F.sigmoid(0.5 * x[:, 7]).unsqueeze(1)], dim=1)
        return x
    
    
def pad_zeros(image, shape):
    result = np.zeros(shape)
    result[:image.shape[0],:image.shape[1]] = image
    return result


def sample_params_string_tab(size):
    stringNumber = np.array([math.floor(np.random.choice(np.arange(0, 6))) for _ in range(size)], dtype=np.int32)
    tab = np.array([math.floor(np.random.choice(np.arange(0, 12))) for _ in range(size)], dtype=np.int32)
#     print(stringNumber)
#     print(tab)
    #  ipdb.set_trace()
    guitars = []
    audio_buffers = []
    cqt_specs = []
    for i in range(size):
        guitars.append(Guitar(options=Options()))
        audio_buffers.append(sequencer.play_note(guitars[i], stringNumber[i], tab[i]))
        cqt_spec = cqt_transform.compute_cqt_spec(audio_buffers[i], n_bins = 336, bins_per_octave=48, hop_length=256).T
        padded_cqt = pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))
        cqt_specs.append(padded_cqt)
    cqt_specs = np.array(cqt_specs, dtype=np.float32)
    print(cqt_specs.shape)
    return stringNumber, tab, cqt_specs
        

def generate_data_string_tab(file, size):
    stringNumber, tab, cqt_specs = sample_params_string_tab(size)
    with open(file, 'wb') as fh:
        data_dict = {'parameters' : np.array([stringNumber, tab]).T, 'cqt_spec' : cqt_specs}
        pkl.dump(data_dict, fh)
    fh.close()
    print(file)
   
    
def sample_params_pitch_sf(size):
    freqs = np.array(utils.compute_freqs(num_frets))
    character_variation = np.array([np.random.uniform(0, 1) for _ in range(size)], dtype=np.float32)
    string_damping = np.array([np.random.uniform(0, 0.7) for _ in range(size)], dtype=np.float32)
    string_damping_variation = np.array([np.random.uniform(0, 0.5) for _ in range(size)], dtype=np.float32)
    pluck_damping = np.array([np.random.uniform(0, 0.9) for _ in range(size)], dtype=np.float32)
    pluck_damping_variation = np.array([np.random.uniform(0, 0.5) for _ in range(size)], dtype=np.float32)
    string_tension = np.array([np.random.uniform(0, 1) for _ in range(size)], dtype=np.float32)
    stereo_spread = np.array([np.random.uniform(0, 1) for _ in range(size)], dtype=np.float32)
    smoothing_factor = np.array([np.random.uniform(0.5, 1) for _ in range(size)], dtype=np.float32)
    pitch = np.array([np.random.choice(freqs) for _ in range(size)], dtype=np.float32)
    #  ipdb.set_trace()
    options = []
    guitars = []
    audio_buffers = []
    cqt_specs = []
    for i in range(size):
        options.append(Options(character_variation[i], string_damping[i], string_damping_variation[i], pluck_damping[i], pluck_damping_variation[i], 
                         string_tension[i], stereo_spread[i]))
        guitars.append(Guitar(options=options[i]))
        audio_buffers.append(sequencer.play_note(guitars[i], 0, 0, pitch[i], smoothing_factor[i]))
#        print(audio_buffers[i])
#         try:
#             cqt_spec = compute_cqt_spec(audio_buffers[i]).T
#         except ParameterError:
#             print(audio_buffers[i])
        cqt_spec = compute_cqt_spec(audio_buffers[i]).T
        padded_cqt = pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))
        cqt_specs.append(padded_cqt)
    cqt_specs = np.array(cqt_specs, dtype=np.float32)
    print(cqt_specs.shape)
    return character_variation, string_damping, string_damping_variation, pluck_damping, pluck_damping_variation, string_tension, stereo_spread, pitch, smoothing_factor, cqt_specs


def save_data_hdf5(fname, size):

    print('Generating {}...'.format(fname))

    freqs = np.array(utils.compute_freqs(num_frets))
    character_variation = np.array([np.random.uniform(0, 1) for _ in range(size)], dtype=np.float32)
    string_damping = np.array([np.random.uniform(0, 0.7) for _ in range(size)], dtype=np.float32)
    string_damping_variation = np.array([np.random.uniform(0, 0.5) for _ in range(size)], dtype=np.float32)
    pluck_damping = np.array([np.random.uniform(0, 0.9) for _ in range(size)], dtype=np.float32)
    pluck_damping_variation = np.array([np.random.uniform(0, 0.5) for _ in range(size)], dtype=np.float32)
    string_tension = np.array([np.random.uniform(0, 1) for _ in range(size)], dtype=np.float32)
    stereo_spread = np.array([np.random.uniform(0, 1) for _ in range(size)], dtype=np.float32)
    smoothing_factor = np.array([np.random.uniform(0.5, 1) for _ in range(size)], dtype=np.float32)
    pitch = np.array([np.random.choice(freqs) for _ in range(size)], dtype=np.float32)

    with h5py.File(fname, 'w') as f:
        dset_parameters = f.create_dataset('parameters', (size, 9), maxshape=(None, None), dtype='float32', chunks=(size, 9))
        dset_cqt_specs = f.create_dataset('cqt_spec', (size, 336, 336), maxshape=(None, None, None), dtype='float32', chunks=(1000, 336, 336))

        dset_parameters[:] = np.array([character_variation, string_damping, string_damping_variation, pluck_damping, pluck_damping_variation, string_tension, stereo_spread, pitch, smoothing_factor]).T

        for i in range(size):
            options = Options(character_variation[i], string_damping[i], string_damping_variation[i], pluck_damping[i], pluck_damping_variation[i], string_tension[i], stereo_spread[i])
            guitar = Guitar(options=options)
            audio_buffer = sequencer.play_note(guitar, 0, 0, pitch[i], smoothing_factor[i])
            cqt_spec = compute_cqt_spec(audio_buffer).T
            padded_cqt = pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))

            dset_cqt_specs[i, :, :] = padded_cqt

    print('Finished generating {}!'.format(fname))


def create_datasets_hdf5(suffix):
    save_data_hdf5('val_{}.h5'.format(suffix), 50)
    save_data_hdf5('eval_{}.h5'.format(suffix), 50)
    save_data_hdf5('test_{}.h5'.format(suffix), 50)
    save_data_hdf5('train_{}.h5'.format(suffix), 100)


def generate_data_pitch_sf(file, size):
    character_variation, string_damping, string_damping_variation, pluck_damping, pluck_damping_variation, string_tension, stereo_spread, pitch, smoothing_factor, cqt_specs = sample_params_pitch_sf(size)
    with open(file, 'wb') as fh:
        data_dict = {'parameters' : np.array([character_variation, string_damping, string_damping_variation, pluck_damping, pluck_damping_variation, string_tension, 
                stereo_spread, pitch, smoothing_factor]).T, 'cqt_spec' : cqt_specs}
        pkl.dump(data_dict, fh)
    fh.close()
    print(file)


def read_data(file):
    with open(file, 'rb') as fh:
        data = pkl.loads(fh.read())
    fh.close()
    return data


def create_datasets(suffix):
    generate_data_pitch_sf("val" + suffix + ".pkl", 500)
#     generate_data('test.pkl', 5000)
#     generate_data('eval.pkl', 5000)
#     generate_data('train.pkl', 50000)
    generate_data_pitch_sf("test" + suffix + ".pkl", 500)
    generate_data_pitch_sf("eval" + suffix + ".pkl", 100)
    generate_data_pitch_sf("train" + suffix + ".pkl", 5000)

    
def read_dataset(suffix):
    #return read_data("train" + suffix + ".pkl"), read_data("test" + suffix + ".pkl"), read_data("val" + suffix + ".pkl"), read_data("eval" + suffix + ".pkl")
    return read_data("val_pitch_sf.pkl"), read_data("test_pitch_sf_sm.pkl"), read_data("val_pitch_sf_sm.pkl"), read_data("eval_pitch_sf.pkl")


def read_data_hdf5(file):
    f = h5py.File(file, 'r')
    dset_parameters = f['parameters']
    dset_cqt_specs = f['cqt_spec']
    return { 'parameters': dset_parameters, 'cqt_spec': dset_cqt_specs }


def read_dataset_hdf5(suffix):
    return (read_data_hdf5('val_{}.h5'.format(suffix)),
            read_data_hdf5('test_{}.h5'.format(suffix)),
            read_data_hdf5('val_{}.h5'.format(suffix)),
            read_data_hdf5('eval_{}.h5'.format(suffix)))


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, parameters, cqt_spectrograms):
        super(MyDataset, self).__init__()
        
        self.parameters = parameters
        self.cqt_spec = cqt_spectrograms
    
    def __getitem__(self, i):
        return self.cqt_spec[i].T, self.parameters[i]
    
    def __len__(self):
        return len(self.parameters)
    
    
def load_data(suffix):
    #create_datasets(suffix)
    #data.generate_data('val.pkl', 5000)
    print("loading data...")
    train_data, test_data, val_data, eval_data = read_dataset(suffix)
    print("data loaded")
    return train_data, test_data, val_data, eval_data


def load_data_hdf5(suffix):
    #create_datasets(suffix)
    #data.generate_data('val.pkl', 5000)
    print("loading data...")
    train_data, test_data, val_data, eval_data = read_dataset_hdf5(suffix)
    print("data loaded")
    return train_data, test_data, val_data, eval_data


def evaluate(net, validation_loader, size, factor):
    criterion = nn.MSELoss()
    val_loss = 0.0
    for i, datapoints in enumerate(validation_loader, 0):

        inputs, labels = datapoints
        inputs.unsqueeze_(1)
        labels = np.delete(labels, 6, axis=1) 
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)
    
        outputs = net(inputs)
        outputs[:, np.arange(6)] = outputs[:, np.arange(6)] * factor
        labels[:, np.arange(6)] =labels[:, np.arange(6)] * factor
        outputs[:, 7] = outputs[:, 7] * factor     
        labels[:, 7] = labels[:, 7] * factor
        loss = criterion(outputs, labels)

        # print statistics
        val_loss += loss.item()
    #return val_loss/float(len(validation_loader.dataset))
    return val_loss/size


def train_model(net, train_data, val_data, eval_data, batch_size, epochs, suffix, trainsize, valsize, factor):
    print("===============Training Data===============")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)
    net.train()
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = MyDataset(parameters=train_data['parameters'], cqt_spectrograms=train_data['cqt_spec'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    valset = MyDataset(parameters=val_data['parameters'], cqt_spectrograms=val_data['cqt_spec'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    evalset = MyDataset(parameters=eval_data['parameters'], cqt_spectrograms=eval_data['cqt_spec'])
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        #net.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            #print(inputs.shape)
            #print(labels.shape)
            inputs.unsqueeze_(1)
            labels = np.delete(labels, 6, axis=1)
            inputs = inputs.to(device)
            labels = labels.to(device)		
            #print(inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            outputs[:, np.arange(6)] = outputs[:, np.arange(6)] * factor
            labels[:, np.arange(6)] =labels[:, np.arange(6)] * factor
            outputs[:, 7] = outputs[:, 7] * factor     
            labels[:, 7] = labels[:, 7] * factor
            #print(outputs[:, 0])
            #print(outputs)
            #m = nn.Sigmoid()
            #outputs = m(5 * (outputs - 0.5))
            #out1 = F.sigmoid(5 * (outputs[:, np.arange(7)] - 0.5))
            #outputs = np.c_[x1.detach().cpu().numpy(), x[:, 7].detach().cpu().numpy()]).to(device)
            loss = criterion(outputs, labels.float())
            #print(loss)
            loss.backward()
            #print("gradients:\n")
            #for param in net.parameters():
            #    print(param.grad)
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
#            if i % 500 == 0:    # print every 20 mini-batches
#                print('[%d, %5d] loss: %.3f' %
#                      (epoch + 1, i + 1, running_loss/500))
#                running_loss = 0.0
            
        print('epoch %d train_loss: %.6f' % (epoch + 1, running_loss/float(len(trainloader.dataset))))
        with open("train_losses" + suffix + ".txt", "a") as text_file:
            #text_file.write(str(running_loss/float(len(trainloader.dataset))))
            text_file.write(str(running_loss/float(trainsize)))
            text_file.write("\n")
            
        val_loss = evaluate(net, valloader, valsize, factor)
        print('epoch %d val_loss: %.6f' % (epoch + 1, val_loss))
        with open("val_losses" + suffix + ".txt", "a") as text_file:
            text_file.write(str(val_loss))
            text_file.write("\n")
    torch.save(net.state_dict(), "checkpoint" + suffix + ".pt")
                    
    print('Finished Training')
    
    
def merge_images(sources, targets, k=10):
    _, h, w = sources.shape
    print(sources.shape[0], h, w)
    row = int(np.sqrt(sources.shape[0]))  # Square root of batch size
    merged = np.zeros([row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    return merged


def test_string_tab(net, test_data):
    net.load_state_dict(torch.load('2fac_checkpoint.pt'))
    net.eval()
    criterion = nn.MSELoss()

    testset = MyDataset(parameters=test_data['parameters'], cqt_spectrograms=test_data['cqt_spec'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    inputs, targets = iter(testloader).next()
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    gt_samples = []
    gt_stringNumbers = []
    gt_tabs = []
    
    for i in range(len(targets)):
        gt_stringNumber, gt_tab = targets.cpu().numpy()[i]
        guitar = Guitar(options=Options())
        gt_stringNumbers.append(gt_stringNumber)
        gt_tabs.append(gt_tab)
        #print("gt_stringNumber: %.3f, gt_tab: %.3f" % (gt_stringNumber, gt_tab))
        audio_buffer = sequencer.play_note(guitar, int(round(gt_stringNumber)), int(round(gt_tab)))
        #cqt_spec = cqt_transform.compute_cqt_spec(audio_buffer).T
        #padded_cqt = pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))        
        gt_samples.append(audio_buffer)

    with open("2fac_gt_data.pkl", 'wb') as fh:
        data_dict = {'gt_samples' : np.array(gt_samples), 'gt_stringNumbers' : np.array(gt_stringNumbers), 'gt_tabs' : np.array(gt_tabs), 'gt_cqts' : inputs.cpu().numpy()}
        pkl.dump(data_dict, fh)
    fh.close()

    preds = net(inputs.unsqueeze_(1))
    preds = preds.detach().cpu().numpy()
    
    pred_samples = []
    pred_cqts = []
    pred_stringNumbers = []
    pred_tabs = []
    
    for i in range(preds.shape[0]):
        pred_stringNumber, pred_tab = preds[i]
        guitar = Guitar(options=Options())
        #print("pred_stringNumber: %d, pred_tab: %d" % (int(round(pred_stringNumber)), int(round(pred_tab))))
        pred_stringNumbers.append(pred_stringNumber)
        pred_tabs.append(pred_tab)
        audio_buffer = sequencer.play_note(guitar, int(round(pred_stringNumber)), int(round(pred_tab)))
        cqt_spec = cqt_transform.compute_cqt_spec(audio_buffer, n_bins=336, bins_per_octave=48, hop_length=256).T
        padded_cqt = pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))      
        pred_cqts.append(padded_cqt.T)
        pred_samples.append(audio_buffer)
        
    with open("2fac_pred_data.pkl", 'wb') as fh:
        data_dict = {'pred_samples' : np.array(pred_samples), 'pred_stringNumbers' : np.array(pred_stringNumbers), 'pred_tabs' : np.array(pred_tabs), 'pred_cqts' : pred_cqts}
        pkl.dump(data_dict, fh)
    fh.close()
    
    print('test_loss: %.3f' % evaluate(net, testloader))
    
    
def test_pitch_sf(net, test_data, batch_size, suffix ,testsize, factor):
    net.load_state_dict(torch.load("checkpoint" + suffix + ".pt"))
    net.eval()
    criterion = nn.MSELoss()

    testset = MyDataset(parameters=test_data['parameters'], cqt_spectrograms=test_data['cqt_spec'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    
    inputs, targets = iter(testloader).next()
    targets = np.delete(targets, 6, axis=1)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    gt_samples = []
    gt_character_variations = []
    gt_string_dampings = []
    gt_string_damping_variations = []
    gt_pluck_dampings = []
    gt_pluck_damping_variations = []
    gt_string_tensions = []
    gt_pitches = []
    gt_smoothing_factors = []
    gt_dumping_variations = []
    
    for i in range(len(targets)):
        gt_character_variation, gt_string_damping, gt_string_damping_variation, gt_pluck_damping, gt_pluck_damping_variation, gt_string_tension, gt_pitch, gt_smoothing_factor= targets.cpu().numpy()[i]
        gt_dumping_variations.append(gt_pluck_damping_variation)
        options = Options(gt_character_variation.astype(np.float64), gt_string_damping.astype(np.float64), gt_string_damping_variation.astype(np.float64), gt_pluck_damping.astype(np.float64), gt_pluck_damping_variation.astype(np.float64), gt_string_tension.astype(np.float64))
        guitar = Guitar(options=options)
        gt_character_variations.append(gt_character_variation)
        gt_string_dampings.append(gt_string_damping)
        gt_string_damping_variations.append(gt_string_damping_variation)
        gt_pluck_dampings.append(gt_pluck_damping)
        gt_pluck_damping_variations.append(gt_pluck_damping_variation)
        gt_string_tensions.append(gt_string_tension)
        gt_pitches.append(gt_pitch)
        gt_smoothing_factors.append(gt_smoothing_factor)
        #print("gt_stringNumber: %.3f, gt_tab: %.3f" % (gt_stringNumber, gt_tab))
        audio_buffer = sequencer.play_note(guitar, 0, 0, gt_pitch.astype(np.float64), gt_smoothing_factor.astype(np.float64))
        cqt_spec = compute_cqt_spec(audio_buffer).T
        padded_cqt = pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))  
        gt_samples.append(audio_buffer)
    
    print("gt:\n")
    print(gt_character_variations)   
    print(gt_string_dampings)
    print(gt_string_damping_variations)
    print(gt_pluck_dampings)
    print(gt_pluck_damping_variations)
    print(gt_string_tensions)
    print(gt_pitches)
    print(gt_smoothing_factors)
        

    with open("gt_data" + suffix + ".pkl", 'wb') as fh:
        data_dict = {'gt_samples' : np.array(gt_samples), 'gt_character_variations': np.array(gt_character_variations) , 'gt_string_dampings' : np.array(gt_string_dampings), 'gt_string_damping_variations' : np.array(gt_string_damping_variations), 'gt_pluck_dampings' : np.array(gt_pluck_dampings), 'gt_pluck_damping_variations' : np.array(gt_pluck_damping_variations), 'gt_string_tensions' : np.array(gt_string_tensions), 
                     'gt_pitches' : np.array(gt_pitches), 'gt_smoothing_factors' : np.array(gt_smoothing_factors), 'gt_cqts' : inputs.cpu().numpy()}
        pkl.dump(data_dict, fh)
    fh.close()

    preds = net(inputs.unsqueeze_(1))
    #m = nn.Sigmoid()
    #preds = m(5 * (preds - 0.5))
    preds = preds.detach().cpu().numpy()
    
    pred_samples = []
    pred_cqts = []
    pred_character_variations = []
    pred_string_dampings = []
    pred_string_damping_variations = []
    pred_pluck_dampings = []
    pred_pluck_damping_variations = []
    pred_string_tensions = []
    pred_pitches = []
    pred_smoothing_factors = []
    pred_dumping_variations = []
    
    for i in range(preds.shape[0]):
        pred_character_variation, pred_string_damping, pred_string_damping_variation, pred_pluck_damping, pred_pluck_damping_variation, pred_string_tension, pred_pitch, pred_smoothing_factor = preds[i]
        #options = Options(pred_character_variation.astype(np.float64), pred_string_damping.astype(np.float64), pred_string_damping_variation.astype(np.float64), pred_pluck_damping.astype(np.float64), pred_pluck_damping_variation.astype(np.float64), pred_string_tension.astype(np.float64), pred_stereo_spread.astype(np.float64))
        pred_dumping_variations.append(pred_pluck_damping_variation)
        options = Options(pred_character_variation.astype(np.float64), pred_string_damping.astype(np.float64), pred_string_damping_variation.astype(np.float64), pred_pluck_damping.astype(np.float64), pred_pluck_damping_variation.astype(np.float64), pred_string_tension.astype(np.float64))
        guitar = Guitar(options=options)
        pred_character_variations.append(pred_character_variation)
        pred_string_dampings.append(pred_string_damping)
        pred_string_damping_variations.append(pred_string_damping_variation)
        pred_pluck_dampings.append(pred_pluck_damping)
        pred_pluck_damping_variations.append(pred_pluck_damping_variation)
        pred_string_tensions.append(pred_string_tension)
        pred_pitches.append(pred_pitch)
        pred_smoothing_factors.append(pred_smoothing_factor)
        #print("gt_stringNumber: %.3f, gt_tab: %.3f" % (gt_stringNumber, gt_tab))
        audio_buffer = sequencer.play_note(guitar, 0, 0, pred_pitch.astype(np.float64), pred_smoothing_factor.astype(np.float64))
        #audio_buffer = sequencer.play_note(guitar, 0, 0, pred_pitch, pred_smoothing_factor)
        cqt_spec = compute_cqt_spec(audio_buffer).T
        padded_cqt = pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))      
        pred_cqts.append(padded_cqt.T)
        pred_samples.append(audio_buffer)

    print("predicted:\n")
    print(pred_character_variations)   
    print(pred_string_dampings)
    print(pred_string_damping_variations)
    print(pred_pluck_dampings)
    print(pred_pluck_damping_variations)
    print(pred_string_tensions)
    print(pred_pitches)
    print(pred_smoothing_factors)
    #print(pred_dumping_variations)

        
    with open("pred_data" + suffix + ".pkl", 'wb') as fh:
        data_dict = {'pred_samples' : np.array(pred_samples), 'pred_character_variations': np.array(pred_character_variations) , 'pred_string_dampings' : np.array(pred_string_dampings), 'pred_string_damping_variations' : np.array(pred_string_damping_variations), 'pred_pluck_dampings' : np.array(pred_pluck_dampings), 'pred_pluck_damping_variations' : np.array(pred_pluck_damping_variations), 'pred_string_tensions' : np.array(pred_string_tensions), 
                     'pred_pitches' : np.array(pred_pitches), 'pred_smoothing_factors' : np.array(pred_smoothing_factors), 'pred_cqts' : pred_cqts}
        pkl.dump(data_dict, fh)
    fh.close()
    
    print('test_loss: %.3f' % evaluate(net, testloader, testsize, factor))


if __name__ == '__main__':
    net = Net_pitch_sf().to(device)
    # create_datasets('_pitch_sf_sm')

    #create_datasets_hdf5('pitch_sf_sm')

    train_data, test_data, val_data, eval_data = load_data("_pitch_sf_sm")
    #train_data, test_data, val_data, eval_data = load_data_hdf5("pitch_sf_sm")

    train_model(net, train_data, val_data, eval_data, 32, 100, "_pitch_sf_fac", 5000, 500, 1000)
    test_pitch_sf(net, test_data, 32, "_pitch_sf_fac", 500, 1000)
