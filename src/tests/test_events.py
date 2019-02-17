from pathlib import Path
import configparser

import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to, is_, close_to

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataprocessing import split_data
from dataprocessing import manager
from dataprocessing import event_detection



def test_window():
    ''' Find the event window in a data sample '''
    dataset_file = 'data/test_data/datasets/random_dataset.txt'
    dataset = manager.load_dataset(dataset_file)
    detection_window = 50
    window = 200
    threshold = 0.1
    meta = pd.DataFrame(np.zeros((dataset.shape[0],1)))
    window_dataset, meta = event_detection.create_window_dataset( \
        dataset, meta, detection_window, window, threshold)
    expected = manager.load_dataset('data/test_data/datasets/random_window_dataset.txt')
    assert(np.allclose(window_dataset, expected))
    assert(meta.shape[1]==3)
    assert_that(meta.iloc[0]['breakpoint_0'], equal_to(54))
    assert_that(meta.iloc[0]['breakpoint_1'], equal_to(254))



def compare_data(dataset_orig, meta_orig, dataset, meta, i):
    orig_row = event_detection.get_original_row(i, meta, meta_orig)
    signal = dataset_orig.to_numpy()[orig_row][1:]
    b0 = meta.columns.get_loc('breakpoint_0')
    b1 = meta.columns.get_loc('breakpoint_1')
    breakpoints = [meta.iloc[i][b0], meta.iloc[i][b1]]
    b0 = breakpoints[0]
    b1 = breakpoints[1]
    window = dataset.iloc[i][1:]    
    assert(np.array_equal(signal[b0:b1], window))


def test_mini_dataset_window():
    ''' Create a dog-specific, balanced, windowed, dataset '''
    dataset_file = 'data/test_data/thirty/thirty.txt'
    meta_file = 'data/test_data/thirty/thirty_meta.txt'
    dest = 'data/test_data/thirty'
    label = 'thirty'
    split_data.mini_dataset(dataset_file, meta_file, 12, 0.3333, 0.5, \
        dog='Samson', events_only=True, \
        event_detection_window=50, event_window=1000, event_threshold=0.1, \
        dest=dest, label=label)
    # Load and test
    dataset = manager.load_dataset(dataset_file)
    meta = manager.load_meta(meta_file)  
    dataset_win_file = dest+'/'+label+'_TRAIN.txt'
    meta_win_file = dest+'/'+label+'_TRAIN_meta.txt'
    dataset_win = manager.load_dataset(dataset_win_file)
    meta_win = manager.load_meta(meta_win_file)    
    #event_detection.plot_window_dataset(dataset, meta, dataset_win, meta_win)
    compare_data(dataset, meta, dataset_win, meta_win, i=7)



if __name__ == "__main__":
    test_mini_dataset_window()