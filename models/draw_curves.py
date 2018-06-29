#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 17:03:15 2018

@author: siqihao
"""

import matplotlib.pyplot as plt
import numpy as np


def get_losses(f):
    losses = []
    logfile = open(f,"r")
    for line in logfile:
        losses.append(float(line))
    return losses


def draw():
    train_losses = get_losses("train_losses.txt")
    val_losses = get_losses("val_losses.txt")
    t = np.linspace(0, len(val_losses), len(val_losses))
    plt.plot(t, np.array(train_losses), 'r')
    plt.plot(t, np.array(val_losses), 'b')  
    plt.show()
    

if __name__ == "__main__":
    draw()