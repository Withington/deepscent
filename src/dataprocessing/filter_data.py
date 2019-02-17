#!/usr/bin/env python3

from pathlib import Path
import argparse
import datetime

import numpy as np
import pandas as pd

from dataprocessing import manager

def flatten_dog_behaviour_database(input, target):
    ''' Flatten an input dog behaviour database csv file by creating one 
    row per sample. The input has information about three samples 
    on each row and some rows are information rows, stating the concentration
    of each sample. 
    Write the output to file, at the specified destination. '''

    # Read in the data and remove unneeded rows and columns
    db = manager.load_dog_behaviour_database(input)
    data = db[db['IsInfoRow']==False]
    # Drop unneeded columns
    cols = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,32]
    data = data.drop(data.columns[cols],axis=1)
    data.Run = data.Run.astype(int)
    data.Pass = data.Pass.astype(int)
    data.rename(index=str, columns={'Dog name': 'DogName'}, inplace=True)

    # Create re-shaped data, one row for each sample.
    # Select position 1 samples
    df_pos1 = data
    df_pos1 = df_pos1.drop(['Concentration2', 'Concentration3','TrueClass2', 'TrueClass3', 'DogClassResult2', 'DogClassResult3', 'Result2', 'Result3'], axis=1)
    df_pos1.rename(index=str, columns={'Concentration1': 'Concentration', 'TrueClass1': 'y_true', 'DogClassResult1': 'y_pred', 'Result1': 'Result'}, inplace=True)
    df_pos1['SensorNumber'] = 0
    # Select position 2 samples
    df_pos2 = data
    df_pos2 = df_pos2.drop(['Concentration1', 'Concentration3','TrueClass1', 'TrueClass3', 'DogClassResult1', 'DogClassResult3', 'Result1', 'Result3'], axis=1)
    df_pos2.rename(index=str, columns={'Concentration2': 'Concentration', 'TrueClass2': 'y_true', 'DogClassResult2': 'y_pred', 'Result2': 'Result'}, inplace=True)
    df_pos2['SensorNumber'] = 1
    # Select position 3 samples
    df_pos3 = data
    df_pos3 = df_pos3.drop(['Concentration1', 'Concentration2','TrueClass1', 'TrueClass2', 'DogClassResult1', 'DogClassResult2', 'Result1', 'Result2'], axis=1)
    df_pos3.rename(index=str, columns={'Concentration3': 'Concentration', 'TrueClass3': 'y_true', 'DogClassResult3': 'y_pred', 'Result3': 'Result'}, inplace=True)
    df_pos3['SensorNumber'] = 2
    # Concatenate the three positions and reset the row index labels
    db_flat = pd.concat([df_pos1, df_pos2, df_pos3])
    db_flat.index = range(len(db_flat.index))

    # Save
    if target:
        manager.save_dog_behaviour_flat_db(target, db_flat, verbose=True)

class DroppedRowsInfo:
    ''' Record which rows are dropped '''
    def __init__(self):
        self.single_match = []  # Rows where there was a single dataset row matching the NS row from the database 
        self.two_match = []     # Rows dropped where there were two matching rows and the timestamp was used to figure out which to drop
        self.multi_match = []   # Rows dropped where there were multiple matching rows so all had to be dropped
        self.no_match = [] 

def remove_samples(db_flat_file, dataset_file, meta_file, dest, label, verbose=True):
    ''' Remove samples where the sample pot was not searched. This 
    information is given in the dog behaviour database. In the Excel
    file it was marked 'NS'. In the csv file, the predicted class
    (i.e. the class indicated by the dog's behaviour) is set to 2

    Parameters
    ----------
    db_flat_file: str
        The flattened dog behaviour database csv file
    dataset_file: str
        An array of pressure sensor data. One row per sample
    meta_file: str
        Meta data corresponding to the dataset. One row per sample
    dest: str
        File directory for saving the new dataset and meta data
    label: str
        Label for naming the output dataset and meta data files
    verbose: bool
        Set to True to print out info
    '''
    db_flat = manager.load_dog_behaviour_flat_db(db_flat_file)
    dataset_df = manager.load_dataset(dataset_file)
    meta_df = manager.load_meta(meta_file)
    remove_samples_from_df(db_flat, dataset_df, meta_df, dest, label, verbose)

def remove_samples_from_df(db_flat, dataset_df, meta_df, dest, label, verbose=True):
    ''' Remove samples where the sample pot was not searched. This 
    information is given in the dog behaviour database. In the Excel
    file it was marked 'NS'. In the csv file, the predicted class
    (i.e. the class indicated by the dog's behaviour) is set to 2 

    Parameters
    ----------
    db_flat: DataFrame
        The flattened dog behaviour database
    dataset_df: DataFrame
        An array of pressure sensor data. One row per sample
    meta_df: DataFrame
        Meta data corresponding to the dataset. One row per sample
    dest: str
        File directory for saving the new dataset and meta data
    label: str
        Label for naming the output dataset and meta data files
    verbose: bool
        Set to True to print out info
    '''

    db_ns = db_flat[db_flat['y_pred']==2] # Class 2 - where the dog did not search the sample (e.g. dog behaviour was "NS")
    dataset_shape_orig = dataset_df.shape
    meta_shape_orig = meta_df.shape
    assert(meta_df.shape[0] == dataset_df.shape[0])

    info = DroppedRowsInfo()
    for s in db_ns.itertuples():
        condition = meta_condition(meta_df, s)
        if meta_df[condition].empty:
            info.no_match.append([s.Date, s.DogName, s.Run, s.Pass, s.SensorNumber])
        else:
            assert(meta_df[condition].shape[0] <= 2)
            if meta_df[condition].shape[0] == 1 :
                info.single_match.append(meta_df[condition])
                drop_a_row(meta_df, dataset_df, condition)
            else:
                #We found 2 rows in meta, try to find 2 rows in the db too
                meta_rows = meta_df[condition]
                this_condition = db_condition(db_flat, s)
                db_rows = db_flat[this_condition]
                if not db_rows.shape[0] == 2:
                    drop_all_rows(meta_df, condition, info, dataset_df)
                else:
                    handle_two_rows(meta_rows, meta_df, db_rows, condition, info, dataset_df)

    assert(meta_df.shape[0]==dataset_df.shape[0])
    if verbose:
        print('\nDatabase samples marked as not searched but where no matching dataset row was found:\n', info.no_match)
        print('\nDropped rows where single matching dataset row was found:\n', info.single_match)
        print('\nDropped rows where there were two matching rows and the timestamp was used to figure out which row(s) to drop:\n', info.two_match)
        print('\nDropped rows where there were multiple matching rows and all had to be dropped:\n', info.multi_match)      
        print('\nDataset shape changed from', dataset_shape_orig, 'to', dataset_df.shape)
        print('Meta data shape changed from', meta_shape_orig, 'to', meta_df.shape)
    if dest:
        dest_dataset = dest + '/' + label + '.txt'
        dest_meta = dest + '/' + label + '_meta.txt'
        manager.save_dataset(dest_dataset, dataset_df, verbose=verbose)
        manager.save_meta(dest_meta, meta_df, verbose=verbose)


def db_condition(db_flat, db_row):
    ''' Assemble a condition that can be used to find rows in 
    the flat database that match the input database row '''
    date = db_flat['Date'] == db_row.Date
    dog = db_flat['DogName'] == db_row.DogName
    run = db_flat['Run'] == db_row.Run
    ps = db_flat['Pass'] == db_row.Pass
    sensor = db_flat['SensorNumber'] == db_row.SensorNumber
    condition = date & dog & run & ps & sensor
    return condition

def meta_condition(meta_df, db_row):
    ''' Assemble a condition that can be used to find rows in a 
    the meta data that match the database row '''
    date = meta_df['date'] == db_row.Date
    dog = meta_df['dog'] == db_row.DogName
    run = meta_df['run'] == db_row.Run
    ps = meta_df['pass'] == db_row.Pass
    sensor = meta_df['sensor_number'] == db_row.SensorNumber
    condition = date & dog & run & ps & sensor
    return condition

def drop_a_row(meta_df, dataset_df, condition):
    assert(meta_df[condition].shape[0] == 1)
    label = meta_df[condition].iloc[-1].name
    i = meta_df.index.get_loc(label)
    meta_df.drop(meta_df.index[i], inplace=True)
    dataset_df.drop(dataset_df.index[i], inplace=True)

def drop_this_row(meta_df, dataset_df, meta_df_row):
    i = meta_df.index.get_loc(meta_df_row.Index)
    meta_df.drop(meta_df.index[i], inplace=True)
    dataset_df.drop(dataset_df.index[i], inplace=True)


def drop_all_rows(meta_df, condition, info, dataset_df):
    ''' Where 2 rows in meta match the condition, but there 
    are not 2 corresponding rows in the database, we can't 
    ascertain which meta row to delete so we must delete them all.
    '''
    for m in meta_df[condition].itertuples():
        info.multi_match.append(m)
        drop_this_row(meta_df, dataset_df, m)


def handle_two_rows(meta_rows, meta_df, db_rows, condition, info, dataset_df):
    ''' Handle the situation where the condition matches 2 rows in meta and also
    matches 2 rows in the database. Use the timestamp in meta to work out 
    which row in meta matches which row in the database. 
    '''
    meta_time_0 = meta_rows.iloc[0]['time']
    meta_time_1 = meta_rows.iloc[1]['time']
    time_condition_0 = meta_df['time'] == meta_time_0
    time_condition_1 = meta_df['time'] == meta_time_1
    if (db_rows.iloc[0].y_pred == 2) & (db_rows.iloc[1].y_pred == 2):
        condition_a = condition & time_condition_0
        condition_b = condition & time_condition_1
        info.two_match.append(meta_df[condition_a])
        info.two_match.append(meta_df[condition_b])
        drop_a_row(meta_df, dataset_df, condition_a)
        drop_a_row(meta_df, dataset_df, condition_b)
    elif db_rows.iloc[0].y_pred == 2 :
        if meta_time_0 < meta_time_1 :
            condition_a = condition & time_condition_0
        else:
            condition_a = condition & time_condition_1
        info.two_match.append(meta_df[condition_a])
        drop_a_row(meta_df, dataset_df, condition_a)
    else:
        assert(db_rows.iloc[1].y_pred == 2)
        if meta_time_0 < meta_time_1 :
            condition_b = condition & time_condition_1
        else:
            condition_b = condition & time_condition_0
        info.two_match.append(meta_df[condition_b])
        drop_a_row(meta_df, dataset_df, condition_b)


def main():
    parser = argparse.ArgumentParser(description='Flatten the dog behaviour database. Create an array with one row for each sample.')
    parser.add_argument('input', help='dog behaviour database csv file')
    parser.add_argument('--target', help='target csv file for saving the flattened database', default='')
    args = parser.parse_args()
    flatten_dog_behaviour_database(args.input, args.target)

if __name__ == "__main__":
    main()