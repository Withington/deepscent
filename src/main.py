#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from dataprocessing import class_info
from dataprocessing import import_data
from dataprocessing import filter_data
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
    import_data.create_dataset(args.dest+'/good.pkl', args.dest+'/private_dataset_prefilter.txt', args.max_cols, args.verbose)
    
    # Use the dog behaviour database to remove any data samples where the dog did not search (the database entry was 'NS'
    db_full = args.dest+'/dog_behaviour_database_private.csv'
    db_flat = args.dest+'/dog_behaviour_database_private_flat.csv'
    dataset = args.dest+'/private_dataset_prefilter.txt'
    meta = args.dest+'/private_dataset_prefilter_meta.txt'
    label = 'private_data_all'
    filter_data.flatten_dog_behaviour_database(db_full, db_flat)
    filter_data.remove_samples(db_flat, dataset, meta, args.dest, label)
    
    # Split the dataset into a training set and a test set.
    split=0.2
    dataset = args.dest+'/private_data_all.txt'
    meta = args.dest+'/private_data_all_meta.txt'
    label = 'private_data_all'
    split_data.split(dataset, meta, test_split=split, \
        dest=args.dest, label=label)
    # plot data
    plotting.plot_dataset(args.dest+'/private_data_all_TRAIN.txt')
    plotting.plot_dataset(args.dest+'/private_data_all_TEST.txt')

    # Further split the training dataset into a smaller training set 
    # and a dev (test) set.
    split=0.25
    dataset = args.dest+'/private_data_all_TRAIN.txt'
    meta = args.dest+'/private_data_all_TRAIN_meta.txt'
    label = 'private_data'
    split_data.split(dataset, meta, test_split=split, \
        dest=args.dest, label=label, shuffle=False)
    # plot data
    plotting.plot_dataset(args.dest+'/private_data_TRAIN.txt')
    plotting.plot_dataset(args.dest+'/private_data_TEST.txt')

    # Make a small training dataset for initial testing. 
    # Like GunPoint - 50 train, 150 test.
    # Make it a balanced dataset.
    num_rows = 200
    class_balance = 0.5
    dataset_bal, meta_bal = split_data.create_balanced_dataset(
        dataset, meta, num_rows, class_balance, shuffle=False)
    split=0.75 
    mini_set_dest = args.dest+'/private_mini'
    label = 'private_mini'
    split_data.split_arrays(dataset_bal, meta_bal, split, 
        mini_set_dest, label, stratify=dataset_bal[:,0])
    plotting.plot_dataset(mini_set_dest+'/private_mini_TRAIN.txt')
    plotting.plot_dataset(mini_set_dest+'/private_mini_TEST.txt')


    #todo lmtw make balanced 1 dog sample




if __name__ == "__main__":
    main()