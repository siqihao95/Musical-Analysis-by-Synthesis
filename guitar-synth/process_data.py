#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:24:48 2018

@author: siqihao
"""

import pdb
#import ipdb
import math
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from guitar import Guitar
import sequencer
import sys
sys.path.insert(0, '../models')
import cqt_transform


device = 'cuda'


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
    

def pad_zeros(image, shape):
    result = np.zeros(shape)
    result[:image.shape[0],:image.shape[1]] = image
    return result


def sample_params(size):
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
        

def generate_data(file, size):
    stringNumber, tab, cqt_specs = sample_params(size)
    with open(file, 'wb') as fh:
        data_dict = {'parameters' : np.array([stringNumber, tab]).T, 'cqt_spec' : cqt_specs}
        pkl.dump(data_dict, fh)
    fh.close()
    print(file)
    
    
def read_data(file):
    with open(file, 'rb') as fh:
        data = pkl.loads(fh.read())
    fh.close()
    return data


def create_datasets():
    generate_data('2fac_val.pkl', 100)
#     generate_data('test.pkl', 5000)
#     generate_data('eval.pkl', 5000)
#     generate_data('train.pkl', 50000)
    generate_data('2fac_test.pkl', 100)
    generate_data('2fac_eval.pkl', 100)
    generate_data('2fac_train.pkl', 500)

    
def read_dataset():
    return read_data('2fac_train.pkl'), read_data('2fac_test.pkl'), read_data('2fac_val.pkl'), read_data('2fac_eval.pkl')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, parameters, cqt_spectrograms):
        super(MyDataset, self).__init__()
        
        self.parameters = parameters
        self.cqt_spec = cqt_spectrograms
    
    def __getitem__(self, i):
        return self.cqt_spec[i].T, self.parameters[i]
    
    def __len__(self):
        return len(self.parameters)
    
    
def load_data():
   # create_datasets()
    #data.generate_data('val.pkl', 5000)
    print("loading data...")
    train_data, test_data, val_data, eval_data = read_dataset()
    print("data loaded")
    return train_data, test_data, val_data, eval_data


def evaluate(net, validation_loader):
    criterion = nn.MSELoss()
    val_loss = 0.0
    for i, datapoints in enumerate(validation_loader, 0):

        inputs, labels = datapoints
        inputs.unsqueeze_(1)
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # print statistics
        val_loss += loss.item()
    #return val_loss/float(len(validation_loader.dataset))
    return val_loss/100


def train_model(net, train_data, val_data, eval_data):
    print("===============Training Data===============")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = MyDataset(parameters=train_data['parameters'], cqt_spectrograms=train_data['cqt_spec'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    valset = MyDataset(parameters=val_data['parameters'], cqt_spectrograms=val_data['cqt_spec'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                            shuffle=False, num_workers=2)
    
    evalset = MyDataset(parameters=eval_data['parameters'], cqt_spectrograms=eval_data['cqt_spec'])
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    for epoch in range(200):  # loop over the dataset multiple times
        #net.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            #print(inputs.shape)
            #print(labels.shape)
            inputs.unsqueeze_(1)
	    inputs = inputs.to(device)
	    labels = labels.to(device)		
            #print(inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
#            if i % 500 == 0:    # print every 20 mini-batches
#                print('[%d, %5d] loss: %.3f' %
#                      (epoch + 1, i + 1, running_loss/500))
#                running_loss = 0.0
            
    print('epoch %d train_loss: %.6f' % (epoch + 1, running_loss/float(len(trainloader.dataset))))
    with open("2fac_train_losses.txt", "a") as text_file:
        text_file.write(str(running_loss/float(len(trainloader.dataset))))
        text_file.write("\n")
            
    val_loss = evaluate(net, valloader)
    print('epoch %d val_loss: %.6f' % (epoch + 1, val_loss))
    with open("2fac_val_losses.txt", "a") as text_file:
        text_file.write(str(val_loss))
        text_file.write("\n")
    torch.save(net.state_dict(), '2fac_checkpoint.pt')

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


def test(net, test_data):
    net.load_state_dict(torch.load('checkpoint.pt'))
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
        gt_stringNumber, gt_tab = targets.numpy()[i]
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
        audio_buffer = play_note(guitar, int(round(pred_stringNumber)), int(round(pred_tab)))
        cqt_spec = cqt_transform.compute_cqt_spec(audio_buffer, n_bins=336, bins_per_octave=48, hop_length=256).T
        padded_cqt = pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))      
        pred_cqts.append(padded_cqt.T)
        pred_samples.append(audio_buffer)
        
    with open("2fac_pred_data.pkl", 'wb') as fh:
        data_dict = {'pred_samples' : np.array(pred_samples), 'pred_stringNumbers' : np.array(pred_stringNumbers), 'pred_tabs' : np.array(pred_tabs), 'pred_cqts' : pred_cqts}
        pkl.dump(data_dict, fh)
    fh.close()
    
    print('test_loss: %.3f' % evaluate(net, testloader))
    
    
        
