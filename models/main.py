#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:55:00 2018

@author: siqihao
"""

import sys
sys.path.insert(0, '../data')
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import data
import cnn


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
    train_data, test_data, eval_data = data.read_dataset()
    return train_data, test_data, eval_data
    
    
def train_model(train_data, test_data, eval_data):
    net = cnn.Net().double()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = MyDataset(parameters=train_data['parameters'], cqt_spectrograms=train_data['cqt_spec'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = MyDataset(parameters=test_data['parameters'], cqt_spectrograms=test_data['cqt_spec'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    
    evalset = MyDataset(parameters=eval_data['parameters'], cqt_spectrograms=eval_data['cqt_spec'])
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=4,
                                             shuffle=False, num_workers=2)
        
    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, datapoints in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = datapoints
            inputs.unsqueeze_(1)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
#            print("gradients:\n")
#            for param in net.parameters():
#                  print(param.grad)
#            print("outputs:\n")
#            print(outputs)
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 200 == 1:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
        print('epoch %d loss: %.3f' % (epoch + 1, running_loss))
        with open("losses.txt", "a") as text_file:
            text_file.write(str("%.3f" % running_loss))
            text_file.write("\n")
            
    print('Finished Training')
  
    
if __name__ == "__main__":
    train_data, test_data, eval_data = load_data()
    train_model(train_data, test_data, eval_data)