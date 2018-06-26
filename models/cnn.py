#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 01:52:53 2018

@author: siqihao
"""

import sys
sys.path.insert(0, '../data')
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import pickle as pkl
import ipdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 81 * 81, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 81 * 81)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
    data.create_datasets()
    train_data, test_data, eval_data = data.read_dataset()
    return train_data, test_data, eval_data
    
    
def train_model(train_data, test_data, eval_data):
    net = Net().double()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = MyDataset(parameters=train_data['parameters'], cqt_spectrograms=eval_data['cqt_spec'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = MyDataset(parameters=test_data['parameters'], cqt_spectrograms=eval_data['cqt_spec'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    evalset = MyDataset(parameters=eval_data['parameters'], cqt_spectrograms=eval_data['cqt_spec'])
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('pitch', 'sampling_freq', 'stretch_factor', 'flag')
    
    for epoch in range(2000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data_point in enumerate(evalloader, 0):
            # get the inputs
            inputs, labels = data_point
            inputs.unsqueeze_(1)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 20 == 1:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
    print('Finished Training')
  
    
if __name__ == "__main__":
    train_data, test_data, eval_data = load_data()
    train_model(train_data, test_data, eval_data)

