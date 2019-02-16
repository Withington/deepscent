#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ruptures as rpt

from dataprocessing import manager

def detect_event():
    ''' Detect change point events using the ruptures package. '''
    dataset_file = 'data/test_data/samson_dataset.txt'
    dataset = manager.load_dataset(dataset_file)

    for i in range(dataset.shape[0]):
        signal = dataset.to_numpy()[i][1:]
        model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
        algo = rpt.Window(width=100, model=model, min_size=500, jump=5).fit(signal)
        breakpoints = algo.predict(n_bkps=2)
        print(breakpoints)
        rpt.show.display(signal, breakpoints, breakpoints, figsize=(10, 6))
        plt.suptitle('Sample '+str(i))
        plt.show()
        rpt.show.display(signal, breakpoints, breakpoints, figsize=(10, 6))
        plt.xlim(breakpoints[0]-50, breakpoints[0]+500)
        plt.suptitle('Sample '+str(i))
        plt.show()



def detect_window():
    ''' Detect event windows in data samples and plot results '''
    dataset_file = 'data/test_data/samson_dataset.txt'
    dataset = manager.load_dataset(dataset_file)
    detection_window = 20 #50
    window = 100 #1000
    threshold = 0.1
    for i in range(dataset.shape[0]):
        signal = dataset.to_numpy()[i][1:]
        breakpoints = max_energy_window(signal, detection_window, window, threshold)
        rpt.show.display(signal, breakpoints, breakpoints, figsize=(10, 6))
        plt.suptitle('Sample '+str(i))
        plt.show()
        rpt.show.display(signal, breakpoints, breakpoints, figsize=(10, 6))
        plt.xlim(breakpoints[0], breakpoints[1])
        plt.suptitle('Zoom in on sample '+str(i))
        plt.show()
    

def create_window_dataset(dataset, meta, detection_window, window, threshold=None):
    ''' Find the event window in each sample in the dataset. 
    Return a dataset containing these windows. '''
    n = dataset.shape[0]
    output = np.zeros((n,window+1))
    output[:,0] = dataset.iloc[:,0]
    meta['breakpoint0'] = 0
    meta['breakpoint1'] = 0
    #def f(x):
    #    return find_window(x, detection_window, window, threshold)
    #output[:,1:window+1], meta[0] = np.apply_along_axis(f, 1, dataset.to_numpy()[:,1:])
    for i in range(n):
        output[i,1:window+1], meta.iloc[i]['breakpoint0'] = \
            find_window(dataset.to_numpy()[i,1:], detection_window, window, threshold)
    
    return pd.DataFrame(output), meta


def find_window(signal, detection_window, window, threshold=None):
    ''' Find the event window in the signal and return that window '''
    breakpoints = max_energy_window(signal, detection_window, window, threshold)
    event_window = signal[breakpoints[0]:breakpoints[1]] 
    return event_window, breakpoints[0] 


def plot_windowing(dataset, window_dataset):
    ''' Plot and compare samples from the original dataset to
    those in the window dataset. Where an event data window was
    selected from the original dataset '''
    assert(dataset.shape[0] == window_dataset.shape[0])
    for i in range(dataset.shape[0]):
        __, ax = plt.subplots(2, 1)
        plt.suptitle('Sample: '+str(i)+' class:' +str(dataset.iloc[i][0]))
        axes = dataset.iloc[i][1:].plot(ax=ax[0], \
            label='Full dataset')
        axes.set_xlim(0, dataset.shape[1])
        axes = window_dataset.iloc[i][1:].plot(ax=ax[1], \
            label='Dataset window')
        axes.set_xlim(0, window_dataset.shape[1])
        plt.show()



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


def main():
    ''' Create and plot a windowed dataset '''
    dataset_file = 'data/test_data/samson_dataset.txt'
    dataset = manager.load_dataset(dataset_file)
    print(dataset.shape)
    detection_window = 20 #50
    window = 100 #1000
    threshold = 0.1
    window_dataset = create_window_dataset(dataset, detection_window, window, threshold)
    plot_windowing(dataset, window_dataset)

if __name__ == "__main__":
    main()