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
    import_data.create_dataset(args.dest+'/good.pkl', args.dest+'/private_dataset_full.txt', args.max_cols, args.verbose)
    # Split the dataset into a training set and a test set.
    split=0.2
    split_data.split(args.dest+'/private_dataset_full.txt', \
        args.dest+'/private_dataset_full_meta.txt', test_split=split, \
        dest=args.dest, label='private_total')
    # plot data
    plotting.plot_dataset(args.dest+'/private_total_TRAIN.txt')
    plotting.plot_dataset(args.dest+'/private_total_TEST.txt')

    # Further split the training dataset into a smaller training set 
    # and a dev (test) set, as per the UCR datasets.
    split=0.25
    split_data.split(args.dest+'/private_total_TRAIN.txt', \
        args.dest+'/private_total_TRAIN_meta.txt', test_split=split, \
        dest=args.dest, label='private', shuffle=False)
    # plot data
    plotting.plot_dataset(args.dest+'/private_TRAIN.txt')
    plotting.plot_dataset(args.dest+'/private_TEST.txt')

    # Make a small training dataset for initial testing. Aiming
    # for a similar size to GunPoint 50 train, 150 test.
    mini_set_dest = args.dest+'/private_mini'
    split=0.875 # To get about 200 in a training set
    split_data.split(args.dest+'/private_total_TRAIN.txt', \
        args.dest+'/private_total_TRAIN_meta.txt', test_split=split, \
        dest=mini_set_dest, label='private_mini_total', shuffle=False)
    split=0.75 # To get a train:test set of about 50 train, 150 test.
    split_data.split(mini_set_dest+'/private_mini_total_TRAIN.txt', \
        mini_set_dest+'/private_mini_total_TRAIN_meta.txt', test_split=split, \
        dest=mini_set_dest, label='private_mini_', shuffle=False)
    # plot data
    plotting.plot_dataset(mini_set_dest+'/private_mini_TRAIN.txt')
    plotting.plot_dataset(mini_set_dest+'/private_mini_TEST.txt')

if __name__ == "__main__":
    main()