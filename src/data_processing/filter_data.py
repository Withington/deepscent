#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import datetime

import numpy as np
import pandas as pd

from data_processing import helper

def flatten_dog_behaviour_database(input, target):
    ''' Flatten an input dog behaviour database csv file by creating one 
    row per sample. The input has information about three samples 
    on each row and some rows are information rows, stating the concentration
    of each sample. 
    Write the output to file, at the specified destination. '''

    # Read in the data and remove unneeded rows and columns
    data_input = helper.load_dog_behaviour_database(input)
    data = data_input[data_input['IsInfoRow']==False]
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
    database_df = pd.concat([df_pos1, df_pos2, df_pos3])
    database_df.index = range(len(database_df.index))

    # Save
    if target:
        helper.save_dog_behaviour_flat_db(target, database_df, verbose=True)


def remove_samples(database, dataset, metaset, dest, prefix):
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
    metaset : str
        Meta data corresponding to the dataset. One row per sample
    dest: str
        File directory for saving the new dataset and metaset
    prefix: str
        Prefix for naming the output dataset and metaset
    '''

    database_df = helper.load_dog_behaviour_flat_db(database)
    # Find any rows where y_pred is class 2, this is where the dog did not search the sample (e.g. dog behaviour was "NS")
    database_ns_df = database_df[database_df['y_pred']==2]

    # Load pressure sensor data
    dataset_df = helper.load_dataset(dataset)
    meta_df = helper.load_meta(metaset)
    print(dataset_df.head())
    print(meta_df.head())
    print(dataset_df.shape)
    print(meta_df.shape)
    assert(meta_df.shape[1]==9)
    assert(meta_df.shape[0] == dataset_df.shape[0])

    assert(meta_df.shape[0]==database_df.shape[0])
    for s in database_ns_df.itertuples():
        date = meta_df['date'] == s.Date
        dog = meta_df['dog'] == s.DogName
        run = meta_df['run'] == s.Run
        ps = meta_df['pass'] == s.Pass
        sensor = meta_df['sensor_number'] == s.SensorNumber
        condition = date & dog & run & ps & sensor
        if meta_df[condition].empty:
            print('Did not find raw data:\n',s.Date, ',', s.DogName, ', Run:', s.Run, ', Pass:', s.Pass, ',Sensor:', s.SensorNumber, '\n')
        else:
            print('Found:\n', s.Date, ',', s.DogName, ',', s.Run, ',', s.Pass, ',', s.SensorNumber) 
            print('Found data is:\n', meta_df[condition].head(), '\n')
            print(meta_df[condition].shape[0])
            assert(meta_df[condition].shape[0]==1)
            i = meta_df.index.get_loc(meta_df[condition].iloc[-1].name)
            print('Row:',i)
            meta_df.drop(meta_df.index[i], inplace=True)
            dataset_df.drop(dataset_df.index[i], inplace=True)
 
    assert(meta_df.shape[0]==dataset_df.shape[0])
    if dest:
        dest_dataset = dest + '/' + prefix + 'dataset.txt'
        dest_metaset = dest + '/' + prefix + 'metaset.txt'
        helper.save_dataset(dest_dataset, dataset_df, verbose=True)
        helper.save_meta(dest_metaset, meta_df, verbose=True)


def main():
    parser = argparse.ArgumentParser(description='Flatten the dog behaviour database. Create an array with one row for each sample.')
    parser.add_argument('input', help='dog behaviour database csv file')
    parser.add_argument('--target', help='target csv file for saving the flattened database', default='')
    args = parser.parse_args()
    flatten_dog_behaviour_database(args.input, args.target)

if __name__ == "__main__":
    main()