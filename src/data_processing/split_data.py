#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_processing import helper

def split(dataset_file, meta_file, test_split, dest='', label='', shuffle=True):
    seed = 99
    # Load dataset
    loaded_dataset = helper.load_dataset_as_np(dataset_file)
    loaded_meta = helper.load_meta_as_np(meta_file)
    n = loaded_meta.shape[0]
    assert(n == loaded_dataset.shape[0])
    # Split dataset
    np.random.seed(99)
    dataset_train, dataset_test, meta_train, meta_test = \
        train_test_split(loaded_dataset, loaded_meta, test_size=test_split, shuffle=shuffle, random_state=seed)
    # Save to file
    if dest:
        dataset_train_name = label + 'dataset_train.txt'
        dataset_test_name = label + 'dataset_test.txt'
        meta_train_name = label + 'metaset_train.txt'   
        meta_test_name = label + 'metaset_test.txt'    
        helper.save_dataset_from_np(dest+'/'+dataset_train_name, dataset_train, verbose=True)
        helper.save_dataset_from_np(dest+'/'+dataset_test_name, dataset_test, verbose=True)
        helper.save_meta_from_np(dest+'/'+meta_train_name, meta_train, verbose=True)
        helper.save_meta_from_np(dest+'/'+meta_test_name, meta_test, verbose=True)




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