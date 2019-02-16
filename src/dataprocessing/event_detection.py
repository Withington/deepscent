#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ruptures as rpt

from dataprocessing import manager

   
def get_original_row(i, meta, meta_orig):
    ''' Return the row index in meta_orig that corresponds to row i in meta '''
    filename = meta_orig.filename == meta.iloc[i]['filename'] 
    sensor = meta_orig.sensor_number == meta.iloc[i]['sensor_number']
    condition = filename & sensor
    assert(meta_orig[condition].shape[0] == 1)
    label = meta_orig[condition].iloc[-1].name
    i = meta_orig.index.get_loc(label)
    return i


def plot_window_dataset(dataset_orig, meta_orig, dataset, meta):
    ''' Plot the original signal and the breakpoints '''
    for i in range(dataset.shape[0]):
        orig_row = get_original_row(i, meta, meta_orig)
        signal = dataset_orig.to_numpy()[orig_row][1:]
        breakpoints = [meta.loc[i, 'breakpoint_0'], meta.loc[i, 'breakpoint_1']]
        rpt.show.display(signal, breakpoints, breakpoints, figsize=(10, 6))
        plt.suptitle('Original data. Sample '+str(i))
        plt.show()
        rpt.show.display(signal, breakpoints, breakpoints, figsize=(10, 6))
        plt.xlim(breakpoints[0]-50, breakpoints[1]+50)
        plt.suptitle('Zoom in on original data window '+str(i))
        plt.show()
        dataset.iloc[i][1:].plot(figsize=(10, 6))
        plt.xlim(-50, dataset.shape[1]+50)
        plt.suptitle('The window dataset '+str(i))
        plt.show()


def create_window_dataset(dataset, meta, detection_window, window, threshold=None):
    ''' Find the event window in each sample in the dataset. 
    Return a dataset containing these windows and add the two breakpoints to 
    the meta data '''
    n = dataset.shape[0]
    output = np.zeros((n,window+1))
    output[:,0] = dataset.iloc[:,0]
    meta['breakpoint_0'] = 0
    meta['breakpoint_1'] = 0
    for i in range(n):
        output[i,1:window+1], breakpoints = \
            find_window(dataset.to_numpy()[i,1:], detection_window, window, threshold)
        meta.loc[i, 'breakpoint_0'] = breakpoints[0]
        meta.loc[i, 'breakpoint_1'] = breakpoints[1]
    return pd.DataFrame(output), meta


def find_window(signal, detection_window, window, threshold=None):
    ''' Find the event window in the signal and return that window '''
    breakpoints = max_energy_window(signal, detection_window, window, threshold)
    event_window = signal[breakpoints[0]:breakpoints[1]] 
    return event_window, breakpoints


def max_energy_window(signal, detection_window, window, threshold=None):
    ''' Find the detection_window of maxium energy, or the first detection_window that
    has energy over the given threshold. Return a window which contains 
    this detection_window on its left. '''
    assert(detection_window <= window)
    assert(window <= signal.shape[0])
    n = signal.shape[0]
    max = 0
    t_max = 0
    for t0 in range(0,n-window):
        energy = signal[t0:t0+detection_window].sum()
        if energy > max:
            max = energy
            t_max = t0
            if threshold and (max > threshold*detection_window):
                break
    breakpoints = [t_max, t_max+window]
    return breakpoints

