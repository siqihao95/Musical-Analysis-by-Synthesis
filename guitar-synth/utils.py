#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 01:03:15 2018

@author: siqihao
"""
import math

a0_hz = 27.5
c0_hz = a0_hz * math.pow(2, -9 / 12)

def compute_freq(octave, semitone):
    basicHz = c0_hz * math.pow(2, octave + semitone / 12)
    return basicHz


def compute_basicHz():
    basicHz = []
    basicHz.append(c0_hz * math.pow(2, 2 + 4 / 12))  #E2
    basicHz.append(c0_hz * math.pow(2, 2 + 9 / 12))  #A2
    basicHz.append(c0_hz * math.pow(2, 3 + 2 / 12))  #D3
    basicHz.append(c0_hz * math.pow(2, 3 + 7 / 12))  #G3
    basicHz.append(c0_hz * math.pow(2, 3 + 11 / 12)) #B3
    basicHz.append(c0_hz * math.pow(2, 4 + 4 / 12))  #E4
    return basicHz
    
    
def compute_freqs(num_frets):
    freqs = []
    num_octaves = 2 + num_frets // 12
    num_semitones = num_octaves * 12 + num_frets % 12 + 1
    for i in range(4, num_semitones + 4):
        freqs.append(compute_freq(2, i))
    return freqs

                   
def compute_pitch(stringNumber, fret):
    strings = compute_basicHz()
    return strings[stringNumber] * math.pow(2, fret/12)

                   
def get_string_tab(freq, num_frets):
    l = []
    for i in range(6):
        for j in range(num_frets + 1):
            if math.isclose(compute_pitch(i, j), freq, rel_tol=1e-5):
                   l.append([int(round(i)), int(round(j))])
    return l