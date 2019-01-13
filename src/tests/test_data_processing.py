from pathlib import Path
import datetime
import configparser

import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to

from data_processing import class_info
from data_processing import import_data


def test_position():
    file_name = '2017_11_06-11_42-Rex_1_1_T1.csv'
    p = class_info.position(file_name)
    assert(p == 'T1')


def test_run_and_pass_no():
    file_name = '2017_11_06-11_42-Rex_6_3_T1.csv'
    run_no, pass_no = class_info.run_and_pass_no(file_name)
    assert(run_no == 6)
    assert(pass_no == 3)


def test_timestamp():
    file_name = '2017_11_06-11_42-Rex_6_3_T1.csv'
    timestamp = class_info.timestamp(file_name)
    expected = datetime.datetime(2017, 11, 6, 11, 42)
    assert(timestamp == expected)

def test_alternative_format():
    file_name = '10-07-2017_Spike_2_2_1-123ppm.csv'
    timestamp, name, run_no, pass_no, position = \
        class_info.alternative_format(file_name)
    expected = datetime.datetime(2017, 7, 10, 00, 00)
    assert(timestamp == expected)
    assert(name == 'Spike')
    assert(run_no == 2)
    assert(pass_no == 2)
    assert(position == 'T1')

    file_name = '10-07-2017_Rex_10_1_2-1.1m-newoil123.csv'
    timestamp, name, run_no, pass_no, position = \
        class_info.alternative_format(file_name)
    expected = datetime.datetime(2017, 7, 10, 00, 00)
    assert(timestamp == expected)
    assert(name == 'Rex') 
    assert(run_no == 10)
    assert(pass_no == 1)
    assert(position == 'T2')

    file_name = '10-07-2017_Spike_5_1_0.csv'
    timestamp, name, run_no, pass_no, position = \
        class_info.alternative_format(file_name)
    expected = datetime.datetime(2017, 7, 10, 00, 00)
    assert(timestamp == expected)
    assert(name == 'Spike') 
    assert(run_no == 5)
    assert(pass_no == 1)
    assert(position == 'B')


def test_alt_dir():
    path = Path('alt_format/10-07-2017_Spike_5_1_0.csv')
    assert(class_info.alternative_format_dir(path))


def test_class_info():
    ''' Go through the raw data files, list the "good" files and 
    list the files that need to be skipped. Extract information 
    from the file names of the "good" files. '''
    source = 'data/test_data/raw_data'
    good, skipped = class_info.parse_filenames(source)
    assert_that(good.shape, equal_to((8,6)))
    assert_that(skipped.shape, equal_to((6,2)))
    # Test contents of one row.
    file0 = good.at[0,'file']
    assert_that(file0.name, equal_to('2017_11_06-11_52-Rex_1_1_T1_.csv'))
    assert(good.at[0,'dog'] == 'Rex')
    assert(good.at[0,'run'] == 1)
    assert(good.at[0,'pass'] == 1)
    assert(good.at[0,'position'] == 'T1')
    time = good.at[0,'timestamp']
    expected = datetime.datetime(2017, 11, 6, 11, 52)
    assert_that(time, equal_to(expected))


def test_raw_data_size():
    ''' Test the data_size function. '''
    input = 'data/test_data/class_info/good.pkl'
    max = import_data.data_size(input)
    assert(max == 12000)


def test_import_data_save():
    ''' Create a dataset from the raw data files. Check that the created 
    dataset and corresponding meta data file contain the expected data. '''
    regenerate = False # Regenerate data/test_data/class_info/good.pkl
    if regenerate:
        source = 'data/test_data/raw_data'
        dest = 'data/test_data/class_info'
        class_info.parse_filenames(source, dest)
    input = 'data/test_data/class_info/good.pkl'
    target = Path('data/test_data/datasets/test_output_dataset.csv')
    target_meta = Path('data/test_data/datasets/test_output_dataset_meta.csv') 
    cols = 6000
    expected_good = 8
    dataset_shape = import_data.create_dataset(input, target, cols)
    assert_that(dataset_shape, equal_to((expected_good*3,cols+1)))
    # Test meta data
    loaded_meta = np.loadtxt(target_meta, dtype=str, delimiter=',')
    assert_that(loaded_meta[10][0], equal_to('2017_11_06-11_38-Rex_5_2_T3.csv'))
    expected_time = datetime.datetime(2017, 11, 6, 11, 38)
    assert_that(loaded_meta[10][1], equal_to(str(expected_time)))
    assert_that(loaded_meta[10][2], equal_to('Rex'))
    assert_that(loaded_meta[10][3], equal_to('5'))
    assert_that(loaded_meta[10][4], equal_to('2'))
    assert_that(loaded_meta[10][5], equal_to('T3'))
    assert_that(loaded_meta[10][6], equal_to('1'))
    assert_that(loaded_meta[10][7], equal_to('0')) 
    # Test the data
    loaded = np.loadtxt(target, delimiter=',')
    raw_loaded = np.loadtxt(
        Path('data/test_data/raw_data/'+loaded_meta[10][0]), 
        delimiter=',')
    cols = raw_loaded.shape[1]
    assert_that(loaded[10][1:cols+1].all(), equal_to(raw_loaded[1][:cols].all())) 


def compare_data(raw_data_path, dataset_file, meta_file, i=None):
    ''' Compare a row from the dataset against the corresponding raw data file. '''
    # Load dataset
    loaded_dataset = np.loadtxt(dataset_file, delimiter=',')   
    loaded_meta = np.loadtxt(meta_file, dtype=str, delimiter=',')
    # Select a row at random.
    rows = loaded_meta.shape[0]
    if not i:
        i = np.random.randint(0,rows)
    filename = loaded_meta[i][0]
    files = raw_data_path.rglob('**/'+filename)
    sensor_num = int(loaded_meta[i][6])
    # Find the corresponding raw data file and test dataset against it.
    count = 0
    for f in files:
        print('i = ', i)
        print('Testing file', f)
        count = count+1
        assert_that(count, equal_to(1))
        raw_loaded = np.loadtxt(f, delimiter=',')  
        cols = raw_loaded.shape[1]
        assert_that(loaded_dataset[i][1:cols+1].all(), 
            equal_to(raw_loaded[sensor_num][:cols].all())) 


def test_dataset():
    ''' Test a row of the dataset against the corresponding raw data file. '''
    i = 7
    raw_data_path = Path('data/test_data/raw_data')
    dataset_file = Path('data/test_data/datasets/test_output_dataset.csv')
    meta_file = Path('data/test_data/datasets/test_output_dataset_meta.csv')
    compare_data(raw_data_path, dataset_file, meta_file, i)


def test_dataset_random():
    ''' Test a random row of the dataset against the corresponding raw data file. '''
    raw_data_path = Path('data/test_data/raw_data')
    dataset_file = Path('data/test_data/datasets/test_output_dataset.csv')
    meta_file = Path('data/test_data/datasets/test_output_dataset_meta.csv')
    compare_data(raw_data_path, dataset_file, meta_file)


def test_dataset_random_user():
    ''' Test a random row of the dataset against the corresponding raw 
    data file. Use data from the files given in the user config, if one is provided.'''
    # Get directories and files from config in order to test private dataset
    config = configparser.ConfigParser()
    config.optionxform=str
    config_files = ['src/public_config.ini', 'src/private_config.ini', 'src/user_config.ini']
    config.read(config_files)
    raw_data_path = Path(config.get('files', 'raw_data_dir'))
    dataset_file = Path(config.get('files', 'dataset'))
    meta_file = Path(config.get('files', 'meta'))
    compare_data(raw_data_path, dataset_file, meta_file)

