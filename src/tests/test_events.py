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
    print(meta)
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
    compare_data(dataset, meta, dataset_win, meta_win, i=7)


def test_drop_no_event():
    ''' Create a dataset by dropping any samples with no event detected'''
    dataset_file = 'data/test_data/thirty/thirty.txt'
    meta_file = 'data/test_data/thirty/thirty_meta.txt'
    dest = 'data/test_data/event_only'
    label = 'event_only'
    dataset = manager.load_dataset(dataset_file)
    meta = manager.load_meta(meta_file) 
    dataset_win, meta_win = event_detection.create_window_dataset(
        dataset, meta, detection_window=50, window=1000, threshold=0.1, drop=True)

    do_save = False
    if do_save:
        dataset_file = 'data/test_data/thirty/events/events.txt'
        meta_file = 'data/test_data/thirty/events/events_meta.txt'
        manager.save_dataset(dataset_file, dataset_win)
        manager.save_meta(meta_file, meta_win)

    # Expect certain rows to have been dropped
    expected = pd.DataFrame([
        ['2017_08_14-12_12_Samson_12_2_B.csv', 2],
        ['2017_08_28-14_42_Samson2_14_1_T2.csv', 0],
        ['2017_09_04-11_47_Samson_3_1_T3.csv', 0],
        ['2017_10_23-14_57-Samson_15_1_T1.csv', 1]],
        columns=['filename', 'sensor_number'])

    meta_merge = meta.merge(meta_win.drop_duplicates(), on=['filename', 'sensor_number'], 
                   how='left', indicator=True)
    dropped = meta_merge[meta_merge['_merge'] == 'left_only']
    dropped.reset_index(inplace=True)
    assert(expected.equals(dropped[['filename', 'sensor_number']]))


if __name__ == "__main__":
    test_mini_dataset_window()