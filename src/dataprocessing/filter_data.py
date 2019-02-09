#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def remove_samples(database, dataset, meta, dest, label):
    ''' Remove samples where the sample pot was not searched. This 
    information is given in the dog behaviour database. In the Excel
    file it was marked 'NS'. In the csv file, the predicted class
    (i.e. the class indicated by the dog's behaviour) is set to 2 

    Parameters
    ----------
    database : str
        The flattened dog behaviour database csv file
    dataset : str
        An array of pressure sensor data. One row per sample
    meta : str
        Meta data corresponding to the dataset. One row per sample
    dest: str
        File directory for saving the new dataset and meta data
    label: str
        Label for naming the output dataset and meta data files
    '''

    db_flat = manager.load_dog_behaviour_flat_db(database)
    # Find any rows where y_pred is class 2, this is where the dog did not search the sample (e.g. dog behaviour was "NS")
    db_ns = db_flat[db_flat['y_pred']==2]

    # Load pressure sensor data
    dataset_df = manager.load_dataset(dataset)
    meta_df = manager.load_meta(meta)
    dataset_shape_orig = dataset_df.shape
    meta_shape_orig = meta_df.shape
    print(dataset_df.head())
    print(meta_df.head())
    assert(meta_df.shape[1]==9)
    assert(meta_df.shape[0] == dataset_df.shape[0])

    for s in db_ns.itertuples():
        date = meta_df['date'] == s.Date
        dog = meta_df['dog'] == s.DogName
        run = meta_df['run'] == s.Run
        ps = meta_df['pass'] == s.Pass
        sensor = meta_df['sensor_number'] == s.SensorNumber
        condition = date & dog & run & ps & sensor
        if meta_df[condition].empty:
            print('Did not find a data row for:\n',s.Date, ',', s.DogName, ', Run:', s.Run, ', Pass:', s.Pass, ',Sensor:', s.SensorNumber, '\n')
        else:
            print('Found data row for:\n', s.Date, ',', s.DogName, ',', s.Run, ',', s.Pass, ',', s.SensorNumber) 
            print('Found data is:\n', meta_df[condition].head(), '\n')
            assert(meta_df[condition].shape[0] <= 2)
            if meta_df[condition].shape[0] == 1 :
                drop_a_row(meta_df, dataset_df, condition)
            else:
                # We found 2 rows in meta, try to find 2 rows in the db too
                meta_rows = meta_df[condition]
                this_date = db_flat['Date'] == s.Date
                this_dog= db_flat['DogName'] == s.DogName
                this_run= db_flat['Run'] == s.Run
                this_pass= db_flat['Pass'] == s.Pass
                this_sensor = db_flat['SensorNumber'] == s.SensorNumber
                this_condition = this_date & this_dog & this_run & this_pass & this_sensor
                db_rows = db_flat[this_condition]
                assert(db_rows.shape[0] == 2)
                print('Found 2 rows in the dog behaviour database too:')
                for r in db_rows.itertuples():
                    print(r)
                meta_time_0 = meta_rows.iloc[0]['time']
                meta_time_1 = meta_rows.iloc[1]['time']
                time_condition_0 = meta_df['time'] == meta_time_0
                time_condition_1 = meta_df['time'] == meta_time_1
                if (db_rows.iloc[0].y_pred == 2) & (db_rows.iloc[1].y_pred == 2):
                    print('Drop both rows')
                    condition_a = condition & time_condition_0
                    condition_b = condition & time_condition_1
                    drop_a_row(meta_df, dataset_df, condition_a)
                    drop_a_row(meta_df, dataset_df, condition_b)
                elif db_rows.iloc[0].y_pred == 2 :
                    if meta_time_0 < meta_time_1 :
                        condition_a = condition & time_condition_0
                    else:
                        condition_a = condition & time_condition_1
                    drop_a_row(meta_df, dataset_df, condition_a)
                else:
                    assert(db_rows.iloc[1].y_pred == 2)
                    if meta_time_0 < meta_time_1 :
                        condition_b = condition & time_condition_1
                    else:
                        condition_b = condition & time_condition_0
                    drop_a_row(meta_df, dataset_df, condition_b)

    assert(meta_df.shape[0]==dataset_df.shape[0])
    if dest:
        dest_dataset = dest + '/' + label + '.txt'
        dest_meta = dest + '/' + label + '_meta.txt'
        manager.save_dataset(dest_dataset, dataset_df, verbose=True)
        manager.save_meta(dest_meta, meta_df, verbose=True)
    print('Dataset shape changed from', dataset_shape_orig, 'to', dataset_df.shape)
    print('Meta data shape changed from', meta_shape_orig, 'to', meta_df.shape)


def drop_a_row(meta_df, dataset_df, condition):
    assert(meta_df[condition].shape[0] == 1)
    i = meta_df.index.get_loc(meta_df[condition].iloc[-1].name)
    print('Dropping row:\n', meta_df[condition])
    meta_df.drop(meta_df.index[i], inplace=True)
    dataset_df.drop(dataset_df.index[i], inplace=True)


def main():
    parser = argparse.ArgumentParser(description='Flatten the dog behaviour database. Create an array with one row for each sample.')
    parser.add_argument('input', help='dog behaviour database csv file')
    parser.add_argument('--target', help='target csv file for saving the flattened database', default='')
    args = parser.parse_args()
    flatten_dog_behaviour_database(args.input, args.target)

if __name__ == "__main__":
    main()