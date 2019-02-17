#!/usr/bin/env python3

from pathlib import Path
import argparse
import datetime

import numpy as np
import pandas as pd

from dataprocessing import manager

def data_size(raw_data_files, verbose=False):
    ''' Input a pkl file listing all of the good raw data files. 
    For each file, count the number of pressure sensor data points. 
    Print summary statistics and return the max number of data 
    points in any file. 
    Return the maximum number of columns in any file'''
    good = pd.read_pickle(Path(raw_data_files))
    n = good.shape[0]
    if verbose:
        print('Number of input files:', n)
    max = 0
    max_file = ''
    size_info = []
    for i in range(0,n):
        file = good.at[i,'file']
        raw_data = manager.load_raw_data_as_np(file)
        num_data_points = raw_data.shape[1]
        size_info.append((file, raw_data.shape[0], num_data_points))
        if num_data_points > max:
            max = num_data_points
            max_file = file
        if verbose and num_data_points >= 30000:
            print(num_data_points, 'data points in file', file)

    if verbose:
        print('max size is', max, 'in file', max_file)
        df = pd.DataFrame(size_info, columns=['file', 'num_samples', 'num_data_points'])
        print('Summary statistics for the number of pressure sensor data points in all of the good files:')
        print(df.num_data_points.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]))
        print('Summary statistics for the number of sample rows in all of the good files:')
        print(df.num_samples.describe())  
        print('Example data:')
        print(df.head())
    return max


def create_dataset(raw_data_files, target='', num_datapoints=None, verbose=False):
    ''' Given a list of raw data files, create a dataset of raw data
    where each sample has the specified number of datapoints. The first
    column of the dataset is the class of each sample.
    Save the entire dataset in the target file. 
    Return the shape of the dataset. '''

    print('Loading', raw_data_files)
    good = pd.read_pickle(Path(raw_data_files))

    max_cols = data_size(raw_data_files, verbose)
    if not num_datapoints:
        num_datapoints = max_cols

    n = good.shape[0]
    class_cols = 1 # How many columns will be added to hold class data.

    data = np.empty((n*3,num_datapoints+class_cols))
    meta = list()
    for i in range(n):
        file = good.at[i,'file']
        d_i = manager.load_raw_data_as_np(file)
        # Set the number of columns by truncating or padding with zeros.
        cols = d_i.shape[1]
        if cols > num_datapoints:
            d_i = d_i[:,:num_datapoints]
        elif cols < num_datapoints:
            d_i = np.pad(d_i,((0,0),(0,num_datapoints-cols)),mode='constant',constant_values=0)
        # Pop class data into the first column.
        position = good.at[i,'position']
        classes_i = class_vector(position)
        d_i = np.hstack((classes_i,d_i))
        # Add this data to the data set.
        data[i*3:i*3+3] = d_i
        # Get the meta data and create three rows of it.
        time_stamp = good.at[i,'timestamp']
        meta_0 = [good.at[i,'file'].name, \
                    time_stamp.date(), \
                    time_stamp.time(), \
                    good.at[i,'dog'], \
                    good.at[i,'run'], \
                    good.at[i,'pass'], \
                    good.at[i,'position'],
                    0,                  # sensor number
                    classes_i[0][0] ]   # class 
        meta.append(meta_0)
        meta_1 = list(meta_0)
        meta_1[7] = 1                   # sensor number
        meta_1[8]  = classes_i[1][0]    # class 
        meta.append(meta_1)
        meta_2 = list(meta_0)
        meta_2[7] = 2                   # sensor number
        meta_2[8] = classes_i[2][0]     # class
        meta.append(meta_2)

    if target:
        output_file = Path(target)
        output_file_meta = Path(str(output_file.parent) + \
            '/' + output_file.stem + '_meta.txt')
        manager.save_dataset_from_np(output_file, data, verbose=True)
        manager.save_meta_from_np(output_file_meta, np.array(meta), verbose=True)

    print('Number of time series data points used:', num_datapoints)
    print('Dataset shape:', data.shape) 
    return data.shape

def class_vector(position):
    dict = {
        'T1': np.array(([1],[0],[0])),
        'T2': np.array(([0],[1],[0])),
        'T3': np.array(([0],[0],[1])),
        'B': np.array(([0],[0],[0]))
    }
    return dict[position]


def main():
    parser = argparse.ArgumentParser(description='Read the raw data files and create a single dataset')
    parser.add_argument('input', help='input file - a pkl file listing all of the files')
    parser.add_argument('--target', help='target txt file for saving the dataset', default='')
    parser.add_argument('--num_datapoints', type=int, help='maximum number of timeseries datapoints to use', default=None)   
    parser.add_argument('--verbose', type=bool, help='print information about the files', default=False)   
    args = parser.parse_args()
    create_dataset(args.input, args.target, args.num_datapoints, args.verbose)

if __name__ == "__main__":
    main()