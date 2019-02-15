#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import configparser

from dataprocessing import class_info
from dataprocessing import import_data
from dataprocessing import filter_data
from dataprocessing import split_data
from dataprocessing import plotting


def main():
    parser = argparse.ArgumentParser(description='Filter the raw data files and create a single dataset')
    parser.add_argument('source', help='source directory containing the raw csv files')
    parser.add_argument('--dest', help='destination file path to save the dataset and intermediate files', default='')
    parser.add_argument('--verbose', type=bool, help='print information about the files', default=False)   
    args = parser.parse_args()
    create_dataset(args)


def create_dataset(args):
    label = 'private_data_all'
    output_dest = args.dest+'/'+label 
    # Get meta data from file names. Discard some files based on their name.
    class_info.parse_filenames(args.source, args.dest)
    # From the 'good' files, get the pressure sensor data and save it all as one dataset.
    num_datapoints = None # Do not truncate data
    import_data.create_dataset(args.dest+'/good.pkl', output_dest+'/private_dataset_prefilter.txt', num_datapoints, args.verbose)
    
    # Use the dog behaviour database to remove any data samples where the dog did not search (the database entry was 'NS'
    db_full = args.dest+'/dog_behaviour_database_private.csv'
    db_flat = args.dest+'/dog_behaviour_database_private_flat.csv'
    dataset_prefilter = output_dest+'/private_dataset_prefilter.txt'
    meta_prefilter = output_dest+'/private_dataset_prefilter_meta.txt'
    filter_data.flatten_dog_behaviour_database(db_full, db_flat)
    filter_data.remove_samples(db_flat, dataset_prefilter, meta_prefilter, output_dest, label)
    os.remove(dataset_prefilter)
    os.remove(meta_prefilter)

    # Split the dataset into a training set and a test set.
    split=0.2
    dataset = output_dest+'/private_data_all.txt'
    meta = output_dest+'/private_data_all_meta.txt'
    split_data.split(dataset, meta, test_split=split, \
        dest=output_dest, label=label)
    # plot data
    plotting.plot_dataset(output_dest+'/private_data_all_TRAIN.txt')
    plotting.plot_dataset(output_dest+'/private_data_all_TEST.txt')


def create_dev_datasets(args):
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

    # Make a mini, balanced dataset of single dog data.
    config = configparser.ConfigParser()
    config.optionxform=str
    config_files = ['src/public_config.ini', 'src/private_config.ini']
    config.read(config_files)
    dog = config._sections['unique_dog_names']['dog2']
    num_rows = 200
    test_split=0.75 
    mini_set_dest = 'data/private_data/private_mini_dog2'
    label = 'private_mini_dog2'
    split_data.mini_dataset(
        dataset, meta, dog, num_rows, test_split, mini_set_dest, label)  
    plotting.plot_dataset(mini_set_dest+'/'+label+'_TRAIN.txt')
    plotting.plot_dataset(mini_set_dest+'/'+label+'_TEST.txt')


if __name__ == "__main__":
    main()