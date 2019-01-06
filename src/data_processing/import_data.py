#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

def data_size(input):
    ''' Input a pkl file listing all of the good raw data files. 
    For each file, count the number of pressure sensor data points. 
    Print summary statistics and return the max number of data 
    points in any file. '''
    good = pd.read_pickle(Path(input))
    n = good.shape[0]
    print('number of good_files', n)
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

def import_data(input, target='', max_cols=11000):
    ''' Read in the list of good files. For each file, get the three pressure
    sensor data samples, add class information (e.g. +ve or -ve scent sample) 
    to the first column(s). Save the entire dataset in a csv file. '''
    save = False
    if target:
        save = True
        output_file = Path(target)
    good = pd.read_pickle(Path(input))
    n = good.shape[0]
    class_cols = 1 # How many columns will be added to hold class data.
    print('Number of raw data input files:', n)

    data = np.empty((n*3,max_cols+class_cols))
    for i in range(0,n):
        file = good.at[i,'file']
        d_i = np.loadtxt(file,delimiter=',')
        assert(d_i.shape[0]==3)
        assert(d_i.shape[1]>100)
        # Set the number of columns by truncating or padding with zeros.
        cols = d_i.shape[1]
        print(i)
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

    print(data.shape)
    if save:
        print('Saving data to', output_file)
        np.savetxt(output_file, data, delimiter=',')

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
    parser.add_argument('input', help='input file path - a pkl file listing all of the files')
    parser.add_argument('--target', help='target file path to save the dataset to csv file', default='')
    parser.add_argument('--max_cols', type=int, help='maximum number of timeseries datapoints to use', default='')   
    args = parser.parse_args()
    import_data(args.input, args.target, args.max_cols)

if __name__ == "__main__":
    main()