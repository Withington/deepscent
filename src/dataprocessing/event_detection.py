#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ruptures as rpt

from . import manager


def get_original_row(i, meta, meta_orig):
    ''' Return the row index in meta_orig that corresponds to row i in meta '''
    filename = meta_orig.filename == meta.iloc[i]['filename'] 
    sensor = meta_orig.sensor_number == meta.iloc[i]['sensor_number']
    condition = filename & sensor
    assert(meta_orig[condition].shape[0] == 1), \
        f'Could not find {filename} and sensor {sensor} in the original meta data file.'
    label = meta_orig[condition].iloc[-1].name
    i = meta_orig.index.get_loc(label)
    return i


def plot_window_dataset(dataset_orig, meta_orig, dataset, meta, index=None):
    ''' Plot the original signal and the breakpoints.
    Plot the windowed dataset
    '''
    start = index if index else 0
    end = index+1 if index else dataset.shape[0]
    for i in range(start, end):
        orig_row = get_original_row(i, meta, meta_orig)
        signal = dataset_orig.to_numpy()[orig_row][1:]
        breakpoints = [meta.iloc[i, 'breakpoint_0'], meta.iloc[i, 'breakpoint_1']]
        rpt.show.display(signal, breakpoints, breakpoints, figsize=(10, 6))
        plt.suptitle('Original data. Sample '+str(i))
        plt.ylim(0, 2.5)
        plt.show()
        rpt.show.display(signal, breakpoints, breakpoints, figsize=(10, 6))
        plt.xlim(breakpoints[0]-50, breakpoints[1]+50)
        plt.ylim(0, 2.5)
        plt.suptitle('Zoom in on original data window '+str(i))
        plt.show()
        dataset.iloc[i][1:].plot(figsize=(10, 6))
        plt.xlim(-50, dataset.shape[1]+50)
        plt.ylim(0, 2.5)
        plt.suptitle('The window dataset '+str(i))
        plt.show()


def create_window_dataset(
        dataset, meta, detection_window, window, threshold=None, drop=False):
    ''' Find the event window in each sample in the dataset.
    Return a dataset containing these windows and add the two breakpoints to
    the meta data.
    Drop rows with no event, if drop is True '''
    n = dataset.shape[0]
    output = np.zeros((n, window+1))
    output[:, 0] = dataset.iloc[:, 0]
    meta_win = pd.DataFrame(meta)
    meta_win['breakpoint_0'] = 0
    meta_win['breakpoint_1'] = 0
    b0 = meta_win.columns.get_loc('breakpoint_0')
    b1 = meta_win.columns.get_loc('breakpoint_1')
    no_event = list()
    for i in range(n):
        output[i, 1:window+1], breakpoints, has_event = \
            find_window(dataset.to_numpy()[i, 1:], 
                        detection_window, window, threshold)
        meta_win.iloc[i, b0] = breakpoints[0]
        meta_win.iloc[i, b1] = breakpoints[1]
        if not has_event:
            no_event.append(i)

    output = pd.DataFrame(output)
    meta_win.reset_index(inplace=True, drop=True)
    if drop:
        output.drop(no_event, axis=0, inplace=True)
        meta_win.drop(no_event, axis=0, inplace=True)
    return output, meta_win


def find_window(signal, detection_window, window, threshold=None):
    ''' Find the event window in the signal and return that window '''
    breakpoints, has_event = max_energy_window(
        signal, detection_window, window, threshold)
    event_window = signal[breakpoints[0]:breakpoints[1]] 
    return event_window, breakpoints, has_event


def max_energy_window(signal, detection_window, window, threshold=None):
    ''' Find the detection_window of maxium energy, or the first
    detection_window that has energy over the given threshold.
    Return a window which contains this detection_window on its left.
    Return has_event = True if there is a detection window that
    exceeds the threshold.
    '''
    assert(detection_window <= window)
    assert(window <= signal.shape[0])
    n = signal.shape[0]
    max = 0
    t_max = 0
    has_event = False
    for t0 in range(0, n-window):
        energy = signal[t0:t0+detection_window].sum()
        if energy > max:
            max = energy
            t_max = t0
            if threshold and (max > threshold*detection_window):
                has_event = True
                break
    breakpoints = [t_max, t_max+window]
    return breakpoints, has_event


def main():
    parser = argparse.ArgumentParser(description='Plot the original signal and the \
        breakpoints. Plot the windowed dataset')
    parser.add_argument('dataset', help='input path to a dataset txt file')
    parser.add_argument('window', help='input path to a windowed dataset txt \
        file')
    parser.add_argument('--index', type=int, help='index of the sample of interest. \
        Plot all if not set', default=None)
    args = parser.parse_args()
    meta = Path(
        Path(args.dataset).parent, (Path(args.dataset).stem + '_meta.txt'))
    window_meta = Path(
        Path(args.window).parent, (Path(args.window).stem + '_meta.txt'))
    # Load data
    dataset = manager.load_dataset(args.dataset)
    meta = manager.load_meta(meta)
    window_dataset = manager.load_dataset(args.window)
    window_meta = manager.load_meta(window_meta)
    plot_window_dataset(dataset, meta, window_dataset, window_meta, args.index)


if __name__ == "__main__":
    main()
