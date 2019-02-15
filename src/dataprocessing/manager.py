#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd


# Raw pressure sensor data (three samples per file) ------------------------------------------------------------------------------------------

def load_raw_data(file):
    ''' Load raw pressure sensor data from csv file and
    return pandas dataframe '''  
    df = pd.read_csv(Path(file), header=None)
    assert(df.shape[0] == 3), ['Expected 3 rows, got', df.shape[0]]
    assert(df.shape[1]>50), ['Expected at least 50 columns, got', df.shape[1]] # Crude check that this looks like a pressure sensor file      
    return df


def load_raw_data_as_np(file):
    ''' Load raw pressure sensor data from csv file and
    return numpy array '''
    array = np.loadtxt(Path(file),  delimiter=',')
    assert(array.shape[0]==3), ['Expected 3 rows, got', array.shape[0]]
    assert(array.shape[1]>50), ['Expected at least 50 columns, got', array.shape[1]] # Crude check that this looks like a pressure sensor file      
    return array

# Dog behaviour database ------------------------------------------------------------------------------------------

def dog_behaviour_database_header():
    header = ['Date', 'Dog name', 'Run', 'Pass', 'Target position', 'Concen-tration', 'Pos 1. sample', \
        'Pos 2. sample', 'Pos 3. sample', 'Pos. 1 behaviour', 'Pos. 2 behaviour', 'Pos. 3 behaviour', \
        'Dog was correct?', 'Comments', 'Pos 1. concentration', 'Pos 2. concentration', \
        'Pos 3. concentration', 'IsInfoRow', 'Corresponding Info Row', 'Concentration1', 'Concentration2', \
        'Concentration3', 'IsLastPass', 'TrueClass1', 'TrueClass2', 'TrueClass3', 'DogClassResult1', \
        'DogClassResult2', 'DogClassResult3', 'Result1', 'Result2', 'Result3', 'End']
    return header

def load_dog_behaviour_database(file):
    ''' Load original dog behaviour database csv file and return 
    pandas dataframe '''
    df = pd.read_csv(Path(file), parse_dates=['Date'])
    assert(list(df)==dog_behaviour_database_header()), ['Unexpected header in dog behaviour database file:', file]
    return df


# Flattened dog behaviour database ------------------------------------------------------------------------------------------

def dog_behaviour_flat_db_header():
    header = ['Date', 'DogName', 'Run', 'Pass', 'Concentration', 'IsLastPass', 'y_true', 'y_pred', 'Result', 'SensorNumber']
    return header


def load_dog_behaviour_flat_db(file):
    ''' Load dog flattened behaviour database csv file and return 
    pandas dataframe '''
    df = pd.read_csv(Path(file), parse_dates=['Date'])
    assert(list(df)==dog_behaviour_flat_db_header()), ['Unexpected header in flattened dog behaviour database file:', file]
    return df


def save_dog_behaviour_flat_db(target, df, verbose=False):
    file = Path(target)
    assert(list(df)==dog_behaviour_flat_db_header()) , ['Unexpected header in flattened dog behaviour database']
    assert(file.suffix=='.csv'), ['Dog behaviour database file type must be .csv, not', file.suffix]
    df.to_csv(file, index=False)
    if verbose: print('Saved flat dog behaviour database to:', file)

# Dataset ------------------------------------------------------------------------------------------

def load_dataset(file):
    ''' Load dataset from txt file and return pandas dataframe '''
    df = pd.read_csv(Path(file), sep=' ', header=None)
    return df

def load_dataset_as_np(file):
    ''' Load dataset txt file and return numpy array '''
    return np.loadtxt(Path(file))   


def save_dataset(target, df, verbose=False):
    ''' Save dataset dataframe to txt file '''
    file = Path(target)
    assert(file.suffix=='.txt'), ['Dataset file type must be .txt, not', file.suffix]
    if verbose: print('Saving dataset to:', file)
    df.to_csv(file, sep=' ', header=False, index=False)
    if verbose: print('Save completed') 


def save_dataset_from_np(target, array, verbose=False):
    ''' Save dataset numpy array to txt file '''
    file = Path(target)   
    assert(file.suffix=='.txt'), ['Dataset file type must be .txt, not', file.suffix]
    if verbose: print('Saving dataset from np array to:', file)
    np.savetxt(target, array, fmt='%f', delimiter=' ')
    if verbose: print('Save completed')


# Meta data ------------------------------------------------------------------------------------------

def meta_header():
    ''' Return the header row for a meta file '''
    return ['filename', 'date', 'time', 'dog', 'run', 'pass', 'positive_position', 'sensor_number', 'class']


def meta_header_as_str():
    ''' Return the header row for a meta file as a single string, comma separated '''
    return ','.join(meta_header()) 


def load_meta(file):
    ''' Load meta data from txt file and return pandas dataframe '''
    df = pd.read_csv(Path(file), sep=',', parse_dates=['date'])
    assert(list(df)==meta_header()), ['Unexpected header in meta file:', file]
    return df

def load_meta_as_np(file):
    ''' Load meta data from txt file and return pandas dataframe '''
    df = pd.read_csv(Path(file), sep=',', parse_dates=['date'])
    assert(list(df)==meta_header())
    return df.to_numpy()


def save_meta(target, df, verbose=False):
    ''' Save meta data dataframe to txt file '''
    file = Path(target)
    assert(file.suffix=='.txt'), ['Meta file type must be .txt, not', file.suffix]
    assert(list(df)==meta_header())
    df.to_csv(file, index=False)
    if verbose: print('Save meta data to:', file)


def save_meta_from_np(target, array, verbose=False):
    ''' Save meta data numpy array to txt file '''
    file = Path(target)   
    assert(file.suffix=='.txt'), ['Meta file type must be .txt, not', file.suffix]
    assert(array.shape[1]==len(meta_header())), ['Array has',array.shape[1], 'columns but expected', len(meta_header()), 'columns']
    if verbose:
        print('Saving meta data from np array to:', file)
        print('Meta data shape is', array.shape)
    np.savetxt(file, array, header=meta_header_as_str(), comments='', fmt='%s', delimiter=',')


def meta_df_from_np(array):
    ''' Create a pandas DataFrame from a numpy array of meta data '''
    assert(array.shape[1]==len(meta_header())), f'Array has {array.shape[1]} columns but expected {len(meta_header())} columns'
    return pd.DataFrame(array, columns=meta_header())

