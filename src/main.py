#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from dataprocessing import class_info
from dataprocessing import import_data
from dataprocessing import split_data
from dataprocessing import plotting


def main():
    parser = argparse.ArgumentParser(description='Read the raw data files and create a single dataset')
    parser.add_argument('source', help='source directory containing the raw csv files')
    parser.add_argument('--dest', help='destination file path to save the dataset and intermediate files', default='')
    parser.add_argument('--max_cols', type=int, help='maximum number of timeseries datapoints to use', default=12000)   
    parser.add_argument('--verbose', type=bool, help='print information about the files', default=False)   
    args = parser.parse_args()
    # Get meta data from file names. Discard some files based on their name.
    class_info.parse_filenames(args.source, args.dest)
    # From the 'good' files, get the pressure sensor data and save it all as one dataset.
    import_data.create_dataset(args.dest+'/good.pkl', args.dest+'/private_dataset.txt', args.max_cols, args.verbose)
    # Split the dataset into a training set and a test set.
    split=0.2
    split_data.split(args.dest+'/private_dataset.txt', \
        args.dest+'/private_dataset_metaset.txt', test_split=split, \
        dest=args.dest, label='private_')
    # plot data
    plotting.plot_dataset(args.dest+'/private_dataset_train.txt')
    plotting.plot_dataset(args.dest+'/private_dataset_test.txt')

    # Further split the training dataset into a smaller training set 
    # and a dev (test) set, as per the UCR datasets.
    split=0.25
    split_data.split(args.dest+'/private_dataset_train.txt', \
        args.dest+'/private_metaset_train.txt', test_split=split, \
        dest=args.dest, label='ucr_private_', shuffle=False)
    # plot data
    plotting.plot_dataset(args.dest+'/ucr_private_dataset_train.txt')
    plotting.plot_dataset(args.dest+'/ucr_private_dataset_test.txt')

    # Make a small training dataset for initial testing. Aiming
    # for a similar size to GunPoint 50 train, 150 test.
    split=0.875 # To get about 200 in a training set
    split_data.split(args.dest+'/private_dataset_train.txt', \
        args.dest+'/private_metaset_train.txt', test_split=split, \
        dest=args.dest, label='split_one_private_', shuffle=False)
    split=0.75 # To get a train:test set of about 50 train, 150 test.
    split_data.split(args.dest+'/split_one_private_dataset_train.txt', \
        args.dest+'/split_one_private_metaset_train.txt', test_split=split, \
        dest=args.dest, label='split_two_private_', shuffle=False)
    # plot data
    plotting.plot_dataset(args.dest+'/split_two_private_dataset_train.txt')
    plotting.plot_dataset(args.dest+'/split_two_private_dataset_test.txt')

if __name__ == "__main__":
    main()