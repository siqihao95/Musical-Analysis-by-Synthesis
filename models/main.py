#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:55:00 2018

@author: siqihao
"""

import sys
sys.path.insert(0, '../data')
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl
import data
import cnn
import karplus_strong
import cqt_transform

device = 'cuda'


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
    #data.create_datasets()
    #data.generate_data('val.pkl', 5000)
    print("loading data...")
    train_data, test_data, val_data, eval_data = data.read_dataset()
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
    return val_loss/5000
    
    
def train_model(net, train_data, val_data, eval_data):
    print("===============Training Data===============")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = MyDataset(parameters=train_data['parameters'], cqt_spectrograms=train_data['cqt_spec'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    
    valset = MyDataset(parameters=val_data['parameters'], cqt_spectrograms=val_data['cqt_spec'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=32,
                                            shuffle=False, num_workers=2)
    
    evalset = MyDataset(parameters=eval_data['parameters'], cqt_spectrograms=eval_data['cqt_spec'])
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=4,
                                             shuffle=False, num_workers=2)
        
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, datapoints in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = datapoints
            inputs.unsqueeze_(1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
#            if i % 500 == 0:    # print every 20 mini-batches
#                print('[%d, %5d] loss: %.3f' %
#                      (epoch + 1, i + 1, running_loss/500))
#                running_loss = 0.0
                
        #print('epoch %d train_loss: %.3f' % (epoch + 1, running_loss/float(len(trainloader.dataset))))
        print('[epoch %d] train_loss: %.3f' % (epoch + 1, running_loss/50000))
        with open("train_losses.txt", "a") as text_file:
            #text_file.write(str(running_loss/float(len(trainloader.dataset))))
            text_file.write(str(running_loss/50000))
            text_file.write("\n")
    
        val_loss = evaluate(net, valloader)
        print('[epoch %d] val_loss: %.3f' % (epoch + 1, val_loss))
        with open("val_losses.txt", "a") as text_file:
            text_file.write(str(val_loss))
            text_file.write("\n")
            
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
    criterion = nn.MSELoss()

    testset = MyDataset(parameters=test_data['parameters'], cqt_spectrograms=test_data['cqt_spec'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    
    inputs, targets = iter(testloader).next()
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    gt_samples = []
    gt_sampling_freqs = []

    for i in range(len(targets)):
        gt_pitch, gt_sampling_freq, gt_stretch_factor, gt_flag = targets.cpu().numpy()[i]
        # print('GT: pitch: {} | sampling_freq: {} | stretch_factor: {} | flag: {}'.format(
        #       gt_pitch, gt_sampling_freq, gt_stretch_factor, gt_flag))
        string = karplus_strong.my_karplus_strong(gt_pitch, 2 * gt_sampling_freq, gt_stretch_factor, 1)
        sample = string.get_samples()
        cqt_spec = cqt_transform.compute_cqt_spec(sample).T
        padded_cqt = data.pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))
    
        gt_samples.append(sample)
        gt_sampling_freqs.append(gt_sampling_freq)
        
    with open("gt_data.pkl", 'wb') as fh:
        data_dict = {'gt_samples' : np.array(gt_samples), 'gt_sampling_freqs' : gt_sampling_freqs, 'gt_cqts' : inputs.cpu().numpy()}
        pkl.dump(data_dict, fh)
    fh.close()
 
    preds = net(inputs.unsqueeze(1))
    preds = preds.detach().cpu().numpy()

    pred_sampling_freqs = []
    pred_samples = []
    pred_cqts = []

    for i in range(preds.shape[0]):
        pred_pitch, pred_sampling_freq, pred_stretch_factor, pred_flag = preds[i]
        # print('PRED: pitch: {} | sampling_freq: {} | stretch_factor: {} | flag: {}'.format(
        #       pred_pitch, pred_sampling_freq, pred_stretch_factor, pred_flag))
        string = karplus_strong.my_karplus_strong(pred_pitch, 2 * pred_sampling_freq, pred_stretch_factor, 1)
        sample = string.get_samples()
        cqt_spec = cqt_transform.compute_cqt_spec(sample).T
        padded_cqt = data.pad_zeros(cqt_spec, (cqt_spec.shape[1], cqt_spec.shape[1]))
    
        pred_sampling_freqs.append(pred_sampling_freq)
        pred_cqts.append(padded_cqt.T)
        pred_samples.append(sample)
        
    with open("pred_data.pkl", 'wb') as fh:
        data_dict = {'pred_samples' : np.array(pred_samples), 'gt_sampling_freqs' : pred_sampling_freqs, 'pred_cqts' : pred_cqts}
        pkl.dump(data_dict, fh)
    fh.close()
    
    print('test_loss: %.3f' % evaluate(net, testloader))
  

def plot_curves():
    train_losses = []
    logfile = open("train_losses.txt","r")
    for line in logfile:
        train_losses.append(float(line))
    val_losses = []
    logfile = open("val_losses.txt","r")
    for line in logfile:
        val_losses.append(float(line))
    t = np.linspace(0, len(val_losses[10:]), len(val_losses[10:]))
 #   plt.plot(t, np.array(train_losses[10:]), 'r')
 #   plt.plot(t, np.array(val_losses[10:]), 'b')
 #   plt.show()

    
def plot_cqts():
    with open("gt_data", 'rb') as fh:
        gt_data = pkl.loads(fh.read())
    fh.close()
    gt_cqts = gt_data['gt_cqts']
    
    with open("pred_data", 'rb') as fh:
        pred_data = pkl.loads(fh.read())
    fh.close()
    pred_cqts = pred_data['pred_cqts']
    
#    plt.figure(figsize=(16,12))
#    plt.imshow(merge_images(gt_cqts[:16], pred_cqts[:16]))
        
        
if __name__ == "__main__":
    net = cnn.Net().double().to(device)
    train_data, test_data, val_data, eval_data = load_data()
    train_model(net, train_data, val_data, eval_data)
    test(net, test_data)
    #plot_curves()
    #plot_cqts()
