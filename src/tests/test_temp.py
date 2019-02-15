
from pathlib import Path
import datetime
import configparser

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.utils

from hamcrest import assert_that, equal_to, is_, close_to


from dataprocessing import split_data
from dataprocessing import manager
from dataprocessing import event_detection
from dataprocessing import filter_data


def compare_data(raw_data_path, dataset_file, meta_file, i=None):
    ''' Compare a row from the dataset against the corresponding raw data file. 
    
    Parameters
    ----------
    raw_data_path: str
        Path to a directory of raw pressure sensor data csv files.
    dataset_file: str
        An array of pressure sensor data in a txt file. One row per sample
    meta_file: str
        Meta data corresponding to the dataset in a txt file. One row per sample
    i: int
        index of the dataset row to test. Selected at random if None.
    '''
    dataset = manager.load_dataset_as_np(dataset_file)   
    meta = manager.load_meta(meta_file)
    compare_data_arrays(raw_data_path, dataset, meta, i)


def compare_data_arrays(raw_data_path, dataset, meta, i=None):
    ''' Compare a row from the dataset against the corresponding raw data file.
        
    Parameters
    ----------
    raw_data_path: str
        Path to a directory of raw pressure sensor data csv files.
    dataset: numpy array
        An array of pressure sensor data. One row per sample
    meta: numpy array
        Meta data corresponding to the dataset. One row per sample
    i: int
        index of the dataset row to test. Selected at random if None.
    '''
    
    # Select a row in the dataset
    assert(dataset.shape[0] == meta.shape[0])
    rows = meta.shape[0]
    if i != None:
        assert(i<rows)
    else:
        i = np.random.randint(0,rows)
    raw_data_filename = meta.iloc[i]['filename']
    # Find the corresponding raw data file and test dataset against it.
    files = raw_data_path.rglob('**/'+raw_data_filename)
    sensor_num = meta.iloc[i]['sensor_number']
    count = 0
    for f in files:
        print('Testing row', i, 'against', f, 'sensor number', sensor_num)
        count = count+1
        assert_that(count, equal_to(1))
        raw_loaded = manager.load_raw_data_as_np(f)
        compare_cols = min(raw_loaded.shape[1], dataset.shape[1]-1)
        print('Comparing the first', compare_cols, 'columns')
        assert(compare_cols>10)
        assert(np.array_equal(dataset[i][1:compare_cols+1], raw_loaded[sensor_num][:compare_cols]))

    if count == 0:
           print('Raw data file with file name', raw_data_filename, 'not found in path', raw_data_path)
    


def test_split_data_temp():
    ''' Test randomly splitting the dataset and meta data into two sets - training and test sets '''
    dataset_file = 'data/test_data/datasets/test_output_dataset.txt'
    meta_file = 'data/test_data/datasets/test_output_dataset_meta.txt'
    dest = 'data/test_data/datasets'
    label = 'test_output_dataset'
    split_data.split(dataset_file, meta_file, 0.2, dest=dest, label=label)
    expected = Path(dest+'/'+label+'_TRAIN.txt')
    assert_that(expected.exists(), is_(True))
    # Compare to raw data
    raw_data_path = Path('data/test_data/raw_data')
    compare_data(raw_data_path, dest+'/'+label+'_TRAIN.txt', dest+'/'+label+'_TRAIN_meta.txt', i=0)


def test_remove_samples_duplicate_db_rows():
    ''' Test removing NS samples where the dog behaviour database contains 
    more than one row with the same date, dog, run, pass and sensor number '''
    db_flat_file = 'data/test_data/samson/duplicate_test_dog_behaviour_database_flat.csv'
    meta_file = 'data/test_data/samson/duplicate_test_dataset_meta.txt'
    dest = 'data/test_data/samson'
    label = 'duplicate_test_filtered_dataset'
    

    db_flat = manager.load_dog_behaviour_flat_db(db_flat_file)
    meta_np = manager.load_meta_as_np(meta_file)

    # Create a dummy dataset and a log relating the data to the meta file
    log = meta_np
    n = meta_np.shape[0]
    dataset_np = np.array
    dataset_np = np.ones((n,20))
    for j in range(n):
        dataset_np[j] = dataset_np[j] * j
        log[j][1] = j

    # Shuffle, for thorough test
    sklearn.utils.shuffle(dataset_np, meta_np)

    # Do the filtering
    dataset_df = pd.DataFrame(dataset_np)
    meta_df = manager.meta_df_from_np(meta_np)
    filter_data.remove_samples_from_df(db_flat, dataset_df, meta_df, dest, label)
    # Test 
    dataset_file = 'data/test_data/samson/'+label+'.txt'
    meta_file = 'data/test_data/samson/'+label+'_meta.txt'
    print('Testing dataset', dataset_file, 'and', meta_file)

    dataset = manager.load_dataset_as_np(dataset_file)   
    meta_np = manager.load_meta_as_np(meta_file)

    for j in range(meta_np.shape[0]):
        filename = meta_np[j][0]
        d_num = int(dataset[j][0])
        filename_in_log = log[d_num][0]
        assert_that(filename, equal_to(filename_in_log))


# def test_mini_dataset():
#     ''' Test that the dog-specific, balanced dataset is created correctly '''
#     raw_data_path = Path('data/test_data/two_dogs/raw_data')
#     dataset = 'data/test_data/two_dogs/test_filtered_dataset.txt'
#     meta = 'data/test_data/two_dogs/test_filtered_dataset_meta.txt'
#     dest = 'data/test_data/two_dogs'
#     label = 'samson_only'
#     split_data.mini_dataset(dataset, meta, 8, 0.5, 0.5, \
#         dog='Samson', events_only=False, dest=dest, label=label)
#     # Load and test
#     dataset = 'data/test_data/two_dogs/samson_only_TRAIN.txt'
#     meta = 'data/test_data/two_dogs/samson_only_TRAIN_meta.txt'
#     for i in range(0,4):
#         compare_data(raw_data_path, dataset, meta, i=i)
#     dataset = 'data/test_data/two_dogs/samson_only_TEST.txt'
#     meta = 'data/test_data/two_dogs/samson_only_TEST_meta.txt'
#     for i in range(0,4):
#         compare_data(raw_data_path, dataset, meta, i=i)


# # def test_mini_dataset_window():
# #     ''' Create a dog-specific, balanced, windowed, dataset '''
# #     raw_data_path = Path('data/test_data/two_dogs/raw_data')
# #     dataset = 'data/test_data/two_dogs/test_filtered_dataset.txt'
# #     meta = 'data/test_data/two_dogs/test_filtered_dataset_meta.txt'
# #     dest = 'data/test_data/two_dogs'
# #     label = 'samson_events'
# #     split_data.mini_dataset(dataset, meta, 8, 0.5, 0.5, \
# #         dog='Samson', events_only=True, \
# #         event_detection_window=10, event_window=50, event_threshold=0.1, \
# #         dest=dest, label=label)
# #     # Load and test
# #     dataset = 'data/test_data/two_dogs/samson_events_TEST.txt'
# #     meta = 'data/test_data/two_dogs/samson_events_TRAIN_meta.txt'
# #     for i in range(0,4):
# #         compare_data(raw_data_path, dataset, meta, i=i)
# #     dataset = 'data/test_data/two_dogs/samson_events_TEST.txt'
# #     meta = 'data/test_data/two_dogs/samson_events_TEST_meta.txt'
# #     for i in range(0,4):
# #         compare_data(raw_data_path, dataset, meta, i=i)