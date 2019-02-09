#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import datetime

import numpy as np
import pandas as pd

def data_size(input):
    ''' Input a pkl file listing all of the good raw data files. 
    For each file, count the number of pressure sensor data points. 
    Print summary statistics and return the max number of data 
    points in any file. '''
    good = pd.read_pickle(Path(input))
    n = good.shape[0]
    print('number of input files:', n)
    max = 0
    max_file = ''
    size_info = []
    for i in range(0,n):
        file = good.at[i,'file']
        df = pd.read_csv(file, header=None)
        s = df.shape[1]
        size_info.append((file, df.shape[0], df.shape[1]))
        if s > max:
            max = s
            max_file = file
        if s >= 30000:
            print(s, 'data points in file', file)

    print('max size is', max)
    print('in file', max_file)
    df = pd.DataFrame(size_info, columns=['file', 'num_samples', 'num_data_points'])
    print('Summary statistics for the number of pressure sensor data points in all of the good files:')
    print(df.num_data_points.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]))
    print('Summary statistics for the number of sample rows in all of the good files:')
    print(df.num_samples.describe())  
    print('Example data:')
    print(df.head())
    return max


def create_dataset(input, target='', max_cols=12000, verbose=False):
    ''' Read in the list of good files, input in a pkl file. 
    For each file, get the three pressure sensor data samples, 
    add class information (e.g. +ve or -ve scent sample) 
    to the first column(s). Save the entire dataset in
    the target file. 
    Return the shape of the dataset. '''

    print('Loading', input)
    good = pd.read_pickle(Path(input))

    if verbose:
        data_size(input)

    n = good.shape[0]
    class_cols = 1 # How many columns will be added to hold class data.

    data = np.empty((n*3,max_cols+class_cols))
    meta = list()
    meta_header = 'filename,date,time,dog,run,pass,positive_position,sensor_number,class'
    for i in range(n):
        file = good.at[i,'file']
        d_i = np.loadtxt(file,delimiter=',')
        assert(d_i.shape[0]==3)
        assert(d_i.shape[1]>50) # Crude check that this looks like a pressure sensor file
        # Set the number of columns by truncating or padding with zeros.
        cols = d_i.shape[1]
        if cols > max_cols:
            d_i = d_i[:,:max_cols]
        elif cols < max_cols:
            d_i = np.pad(d_i,((0,0),(0,max_cols-cols)),mode='constant',constant_values=0)
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
            '/' + output_file.stem + '_metaset.txt')
        print('Saving data to', output_file, 'and', output_file_meta)
        np.savetxt(output_file, data, fmt='%f', delimiter=' ')
        np.savetxt(output_file_meta, meta, header=meta_header, comments='', fmt='%s', delimiter=',')

    print('Number of time series data points used:', max_cols)
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
    parser.add_argument('--max_cols', type=int, help='maximum number of timeseries datapoints to use', default=12000)   
    parser.add_argument('--verbose', type=bool, help='print information about the files', default=False)   
    args = parser.parse_args()
    create_dataset(args.input, args.target, args.max_cols, args.verbose)

if __name__ == "__main__":
    main()