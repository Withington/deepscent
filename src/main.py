#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from data_processing import class_info
from data_processing import import_data


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
    import_data.create_dataset(args.dest+'/good.pkl', args.dest+'/private_dataset.csv', args.max_cols, args.verbose)


if __name__ == "__main__":
    main()