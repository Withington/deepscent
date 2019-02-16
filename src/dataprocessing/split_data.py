#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.utils

from dataprocessing import manager
from dataprocessing import event_detection

def split(dataset_file, meta_file, test_split, dest='', label='', shuffle=True, stratify=None):
    ''' Split the dataset and corresponding meta into train and test sets.
    Save to dest. '''
    dataset = manager.load_dataset_as_np(dataset_file)
    meta = manager.load_meta_as_np(meta_file)
    split_arrays(dataset, meta, test_split, dest, label, shuffle, stratify)

def split_arrays(dataset, meta, test_split, dest='', label='', shuffle=True, stratify=None):
    ''' Split the dataset and corresponding meta into train and test sets.
    Save to dest. '''
    seed = 99
    np.random.seed(99)
    assert(meta.shape[0] == dataset.shape[0])
    # Split dataset
    dataset_train, dataset_test, meta_train, meta_test = \
        train_test_split(dataset, meta, test_size=test_split, 
        stratify=stratify, shuffle=shuffle, random_state=seed)
    # Save to file
    if dest:
        dataset_train_name = label + '_TRAIN.txt'
        dataset_test_name = label + '_TEST.txt'
        meta_train_name = label + '_TRAIN_meta.txt'   
        meta_test_name = label + '_TEST_meta.txt' 
        print('\ndataset_test:', dataset_test)   # todo lmtw remove
        manager.save_dataset_from_np(dest+'/'+dataset_train_name, dataset_train, verbose=True)
        manager.save_dataset_from_np(dest+'/'+dataset_test_name, dataset_test, verbose=True)
        manager.save_meta_from_np(dest+'/'+meta_train_name, meta_train, verbose=True)
        manager.save_meta_from_np(dest+'/'+meta_test_name, meta_test, verbose=True)


def create_balanced_dataset(dataset, meta, num, class_balance, shuffle=True):
    ''' Create a dataset of the given number of rows and with the given
    class balance between the two classes. Return the balanced dataset 
    and corresponding meta DataFrames. '''
    dataset_bal, meta_bal = create_balanced_dataset_from_arrays(
        dataset.to_numpy(), meta.to_numpy(), num, class_balance, shuffle)
    dataset_bal = pd.DataFrame(dataset_bal)
    meta_bal = manager.meta_df_from_np(meta_bal)    
    return dataset_bal, meta_bal

def create_balanced_dataset_from_arrays(dataset, meta, num, class_balance, shuffle=True):
    ''' Create a dataset of the given number of rows and with the given
    class balance between the two classes. Return the balanced dataset 
    and corresponding meta DataFrames. '''
    assert(class_balance <= 1)
    np.random.seed(99)   
    assert(meta.shape[0] == dataset.shape[0])
    assert(num <= meta.shape[0])

    if shuffle:
        dataset, meta = sklearn.utils.shuffle(dataset, meta)
    
    dataset_bal = np.empty((num,dataset.shape[1]))
    meta_bal = list()
    n0 = round(num*class_balance)
    n1 = num - n0
    c0 = 0
    c1 = 0
    for i in range(0,meta.shape[0]):
        if (c0 + c1) == num:
            break
        y = int(dataset[i][0])
        if y == 0:
            if c0 < n0:
                dataset_bal[c0+c1] = dataset[i]
                meta_bal.append(meta[i])
                c0 = c0 + 1
        else:
            assert(y == 1), 'Dataset must have only two classes - 0 and 1'
            if c1 < n1:
                dataset_bal[c0+c1] = dataset[i]
                meta_bal.append(meta[i])
                c1 = c1 + 1

    assert((c0 + c1) == num), f'A balanced dataset of {num} rows could not be created.'    
    meta_bal = np.array(meta_bal)
    return dataset_bal, meta_bal


def mini_dataset(dataset_file, meta_file, \
        num_samples, test_split, class_balance=0.5, \
        dog=None, events_only=False,
        event_detection_window=50, event_window=1000, event_threshold=0.1, \
        dest=None, label=None):
    ''' Create mini, balanced, dataset, with meta data, for the given dog.
    Save it in dest, using label to name the files '''

    dataset = manager.load_dataset(dataset_file)
    meta = manager.load_meta(meta_file)
    assert(dataset.shape[0] == meta.shape[0])

    # Use data for only one dog
    if dog:
        dataset, meta = dataset_for_dog(dataset, meta, dog)

    # Create a smaller, balanced dataset
    dataset, meta = create_balanced_dataset(
        dataset, meta, num_samples, 
        class_balance, shuffle=False)

    # Reduce samples to the event window
    if events_only:
        dataset, meta = event_detection.create_window_dataset( \
            dataset, meta, event_detection_window, event_window, event_threshold)

    # Split in to training and test sets, maintaining the balanace
    split_arrays(dataset, meta, test_split, 
        dest, label, stratify=dataset[0])


def dataset_for_dog(dataset, meta, dog):
    ''' Return dataset and meta data for the given dog '''
    assert(dataset.shape[0] == meta.shape[0])
    n_meta = meta.shape[1]
    df = pd.concat([meta, dataset], axis=1, join_axes=[meta.index])
    assert(df.shape[0] == meta.shape[0])
    df = df[df.dog == dog]
    dog_meta_df = df.iloc[:,:n_meta]
    dog_df = df.iloc[:,n_meta:]
    return dog_df, dog_meta_df






def main():
    parser = argparse.ArgumentParser(description='Split a dataset into training and test datasets')
    parser.add_argument('input_file', help='input dataset txt file')
    parser.add_argument('input_meta', help='input dataset meta data txt file')    
    parser.add_argument('test_split', type=float, help='proportion of data samples to split out into the test set (0 to 1)')    
    parser.add_argument('--dest', help='destination for saving the training and test datasets and meta data', default='') 
    parser.add_argument('--label', help='label the saved files with this prefix', default='')    
    parser.add_argument('--shuffle', type=bool,  help='shuffle the data before splitting it', default=True)    
    
    args = parser.parse_args()
    split(args.input_file, args.input_meta, args.test_split, args.dest, args.label)


if __name__ == "__main__":
    main()