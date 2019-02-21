#!/usr/bin/env python3

import os
import argparse
import configparser

from dataprocessing import class_info
from dataprocessing import event_detection
from dataprocessing import filter_data
from dataprocessing import import_data
from dataprocessing import manager
from dataprocessing import plotting
from dataprocessing import split_data
from tests import test_events


def create_dataset():
    parser = argparse.ArgumentParser(description='Filter the raw data files and create a single dataset')
    parser.add_argument('source', help='source directory containing the raw csv files')
    parser.add_argument('--dest', help='destination file path to save the dataset and intermediate files', default='')
    parser.add_argument('--verbose', type=bool, help='print information about the files', default=False)
    args = parser.parse_args()

    label = 'private_data_all'
    output_dest = args.dest+'/'+label
    # Get meta data from file names. Discard some files based on their name.
    class_info.parse_filenames(args.source, args.dest)
    # From the 'good' files, get the pressure sensor data and save it all as one dataset.
    num_datapoints = None  # Do not truncate data
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
    split = 0.2
    dataset = output_dest+'/private_data_all.txt'
    meta = output_dest+'/private_data_all_meta.txt'
    split_data.split(dataset, meta, test_split=split,
        dest=output_dest, label=label)
    # plot data
    plotting.plot_dataset(output_dest+'/private_data_all_TRAIN.txt')
    plotting.plot_dataset(output_dest+'/private_data_all_TEST.txt')


def create_window_dataset():
    ''' Remove all samples where no event is detected and create sample windows '''
    parser = argparse.ArgumentParser(description='Remove all samples where no event is detected and create sample windows')
    parser.add_argument('source', help='source directory containing the overall dataset')
    parser.add_argument('--dest', help='destination file path to save the dataset and meta data', default='')
    args = parser.parse_args()

    # Load original dataset
    dataset_file = args.source+'/private_data_all.txt'
    meta_file = args.source+'/private_data_all_meta.txt'
    label = 'private_events_all'
    dataset = manager.load_dataset(dataset_file)
    meta = manager.load_meta(meta_file) 

    # Select event windows, remove samples with no event
    detection_window = 50
    window = 1000
    threshold = 0.1
    dataset_win, meta_win = event_detection.create_window_dataset(
        dataset, meta, detection_window=detection_window, 
        window=window, threshold=threshold, drop=True)
    if args.dest:
        dataset_file = args.dest+'/'+label+'/'+label+'.txt'
        meta_file = args.dest+'/'+label+'/'+label+'_meta.txt'
        manager.save_dataset(dataset_file, dataset_win, verbose=True)
        manager.save_meta(meta_file, meta_win, verbose=True)
    # plot data
    plotting.plot_dataset(dataset_file)

    # Test that window data can be compared back to original data
    n = dataset_win.shape[0]
    test_events.compare_data(dataset, meta, dataset_win, meta_win, i=round(n/3))
    test_events.compare_data(dataset, meta, dataset_win, meta_win, i=round(2*n/3))
    test_events.compare_data(dataset, meta, dataset_win, meta_win, i=n-1)
    print('Test successful')

    # Split the windowed dataset into a training set and a test set.
    split = 0.2
    split_data.split(dataset_file, meta_file, test_split=split,
        dest=args.dest+'/'+label, label=label)
    # plot data
    plotting.plot_dataset(args.dest+'/private_events_all/private_events_all_TRAIN.txt')
    plotting.plot_dataset(args.dest+'/private_events_all/private_events_all_TEST.txt')

    # Further split the training dataset into a smaller training set
    # and a dev (test) set that will be used during development
    split = 0.25
    dataset = args.dest+'/private_events_all/private_events_all_TRAIN.txt'
    meta = args.dest+'/private_events_all/private_events_all_TRAIN_meta.txt'
    label = 'private_events_dev'
    dest = args.dest+'/'+label
    split_data.split(dataset, meta, test_split=split,
        dest=dest, label=label, shuffle=False)
    # plot data
    plotting.plot_dataset(dest+'/private_events_dev_TRAIN.txt')
    plotting.plot_dataset(dest+'/private_events_dev_TEST.txt')


def create_dev_datasets():
    ''' Create focused datasets, for development work '''
    parser = argparse.ArgumentParser(description='Create datasets. Reduce size by selecting an event window from each sample.')
    parser.add_argument('source', help='source directory containing the overall dataset')
    parser.add_argument('--dest', help='destination file path to save the dataset and meta data', default='')
    args = parser.parse_args()
 
    # Make a small training datasets for initial testing.
    dataset = args.source+'/private_events_dev_TRAIN.txt'
    meta = args.source+'/private_events_dev_TRAIN_meta.txt'
    num_samples = 200
    class_balance = 0.5
    split = 0.75


    # Multi-dog set
    label = 'private_mini'
    dest = args.dest+'/'+label
    split_data.mini_dataset(dataset, meta,
        num_samples=num_samples, test_split=split, class_balance=class_balance,
        dest=dest, label=label)
    plotting.plot_dataset(dest+'/private_mini_TRAIN.txt')
    plotting.plot_dataset(dest+'/private_mini_TEST.txt')

    # Single dog set
    config = configparser.ConfigParser()
    config.optionxform = str
    config_files = ['src/public_config.ini', 'src/private_config.ini']
    config.read(config_files)
    dog = config._sections['unique_dog_names']['dog2']
    label = 'private_mini_dog2'
    dest = args.dest+'/'+label
    split_data.mini_dataset(dataset, meta,
        num_samples=num_samples, test_split=split, class_balance=class_balance,
        dog=dog, dest=dest, label=label)
    plotting.plot_dataset(dest+'/'+'private_mini_dog2_TRAIN.txt')
    plotting.plot_dataset(dest+'/'+'private_mini_dog2_TEST.txt')


if __name__ == "__main__":
    #create_dataset()
    #create_window_dataset()
    create_dev_datasets()
