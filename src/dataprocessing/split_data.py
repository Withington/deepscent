#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dataprocessing import manager

def split(dataset_file, meta_file, test_split, dest='', label='', shuffle=True):
    seed = 99
    np.random.seed(99)
    # Load dataset
    dataset_full = manager.load_dataset_as_np(dataset_file)
    meta_full = manager.load_meta_as_np(meta_file)
    assert(meta_full.shape[0] == dataset_full.shape[0])
    # Split dataset
    dataset_train, dataset_test, meta_train, meta_test = \
        train_test_split(dataset_full, meta_full, test_size=test_split, shuffle=shuffle, random_state=seed)
    # Save to file
    if dest:
        dataset_train_name = label + '_TRAIN.txt'
        dataset_test_name = label + '_TEST.txt'
        meta_train_name = label + '_TRAIN_meta.txt'   
        meta_test_name = label + '_TEST_meta.txt'    
        manager.save_dataset_from_np(dest+'/'+dataset_train_name, dataset_train, verbose=True)
        manager.save_dataset_from_np(dest+'/'+dataset_test_name, dataset_test, verbose=True)
        manager.save_meta_from_np(dest+'/'+meta_train_name, meta_train, verbose=True)
        manager.save_meta_from_np(dest+'/'+meta_test_name, meta_test, verbose=True)




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