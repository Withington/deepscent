from pathlib import Path
import datetime
import configparser

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.utils

from hamcrest import assert_that, equal_to, is_, close_to

from dataprocessing import class_info
from dataprocessing import import_data
from dataprocessing import filter_data
from dataprocessing import split_data
from dataprocessing import manager


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
    target = Path('data/test_data/datasets/test_output_dataset.txt')
    target_meta = Path('data/test_data/datasets/test_output_dataset_meta.txt') 
    datapoints = 6000
    expected_good = 8
    dataset_shape = import_data.create_dataset(input, target, datapoints)
    assert_that(dataset_shape, equal_to((expected_good*3,datapoints+1)))
    # Test meta data
    meta = manager.load_meta(target_meta)
    i = 10
    assert_that(meta.iloc[i]['filename'], equal_to('2017_11_06-11_38-Rex_5_2_T3.csv'))
    expected_time = datetime.datetime(2017, 11, 6, 11, 38)
    assert_that(meta.iloc[i]['date'].date(), equal_to(expected_time.date()))
    assert_that(meta.iloc[i]['time'], equal_to(str(expected_time.time())))
    assert_that(meta.iloc[i]['dog'], equal_to('Rex'))
    assert_that(meta.iloc[i]['run'], equal_to(5))
    assert_that(meta.iloc[i]['pass'], equal_to(2))
    assert_that(meta.iloc[i]['positive_position'], equal_to('T3'))
    assert_that(meta.iloc[i]['sensor_number'], equal_to(1))
    assert_that(meta.iloc[i]['class'], equal_to(0)) 
    # Test the data
    loaded = manager.load_dataset_as_np(target)
    raw_filename = meta.iloc[i]['filename']
    raw_loaded = manager.load_raw_data_as_np('data/test_data/raw_data/'+raw_filename)
    cols = raw_loaded.shape[1]
    assert(np.array_equal(loaded[i][1:cols+1], raw_loaded[1]))


def compare_data_files(raw_data_path, dataset_file, meta_file, i='random'):
    ''' Compare a row from the dataset against the corresponding raw data file. 
    
    Parameters
    ----------
    raw_data_path: str
        Path to a directory of raw pressure sensor data csv files.
    dataset_file: str
        An array of pressure sensor data in a txt file. One row per sample
    meta_file: str
        Meta data corresponding to the dataset in a txt file. One row per sample
    i: str
        index of the dataset row to test. Or 'all' or 'random'.
    '''
    dataset = manager.load_dataset(dataset_file)   
    meta = manager.load_meta(meta_file)
    compare_data(raw_data_path, dataset, meta, i)


def compare_data(raw_data_path, dataset, meta, i='random'):
    ''' Compare a row from the dataset against the corresponding raw data file.
        
    The dataset may be truncated or padded compared to the raw data so
    overrunning columns are ignored.

    Parameters
    ----------
    raw_data_path: str
        Path to a directory of raw pressure sensor data csv files.
    dataset: DataFrame
        A pressure sensor dataset. One row per sample
    meta: DataFrame
        Meta data corresponding to the dataset. One row per sample
    i: str
        index of the dataset row to test. Or 'all' or 'random'.
    '''
    
    # Select a row in the dataset
    assert(dataset.shape[0] == meta.shape[0])
    rows = meta.shape[0]
    if i == 'all':
        start = 0
        end = rows
    elif i == 'random':
        start = np.random.randint(0,rows)
        end = start+1
    else:
        start = int(i)
        assert(i<rows)
        end = start+1

    for idx in range(start, end):   
        raw_data_filename = meta.iloc[idx]['filename']
        # Find the corresponding raw data file and test dataset against it.
        files = raw_data_path.rglob('**/'+raw_data_filename)
        sensor_num = meta.iloc[idx]['sensor_number']
        count = 0
        for f in files:
            print('Testing row', idx, 'against', f)
            count = count+1
            assert_that(count, equal_to(1))
            raw_loaded = manager.load_raw_data_as_np(f)
            compare_cols = min(raw_loaded.shape[1], dataset.shape[1]-1)
            print('Comparing the first', compare_cols, 'columns')
            assert(compare_cols>10)
            assert(np.array_equal(dataset.iloc[idx][1:compare_cols+1], raw_loaded[sensor_num][:compare_cols]))


def test_dataset():
    ''' Test a row of the dataset against the corresponding raw data file. '''
    i = 7
    raw_data_path = Path('data/test_data/raw_data')
    dataset_file = Path('data/test_data/datasets/test_output_dataset.txt')
    meta_file = Path('data/test_data/datasets/test_output_dataset_meta.txt')
    compare_data_files(raw_data_path, dataset_file, meta_file, i)


def test_dataset_random():
    ''' Test a random row of the dataset against the corresponding raw data file. '''
    raw_data_path = Path('data/test_data/raw_data')
    dataset_file = Path('data/test_data/datasets/test_output_dataset.txt')
    meta_file = Path('data/test_data/datasets/test_output_dataset_meta.txt')
    compare_data_files(raw_data_path, dataset_file, meta_file)


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
    # Check the test set
    dataset_file = Path(config.get('files', 'dataset_test'))
    meta_file = Path(config.get('files', 'meta_test'))
    dataset = manager.load_dataset(dataset_file)
    meta = manager.load_meta(meta_file)
    n = meta.shape[0]
    compare_data(raw_data_path, dataset, meta, i=0)
    compare_data(raw_data_path, dataset, meta, i=n-1)
    compare_data(raw_data_path, dataset, meta)

def test_split_data():
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
    compare_data_files(raw_data_path, dest+'/'+label+'_TRAIN.txt', dest+'/'+label+'_TRAIN_meta.txt', i=0)


def test_split_data_reload():
    ''' Test the training and test sets against the raw data 
    in the files specified by the public config '''
    config = configparser.ConfigParser()
    config.optionxform=str
    config_files = ['src/public_config.ini']
    config.read(config_files)
    # Load training and test data and compare against original
    i = 2
    raw_data_path = Path(config.get('files', 'raw_data_dir'))
    dataset_file = Path(config.get('files', 'dataset_train'))
    meta_file = Path(config.get('files', 'meta_train'))
    compare_data_files(raw_data_path, dataset_file, meta_file, i)
    compare_data_files(raw_data_path, dataset_file, meta_file) # Random i  
    # Test the _test set
    dataset_file = Path(config.get('files', 'dataset_test'))
    meta_file = Path(config.get('files', 'meta_test'))
    compare_data_files(raw_data_path, dataset_file, meta_file, i)
    compare_data_files(raw_data_path, dataset_file, meta_file)  # Random i 


def test_split_data_user():
    ''' Test the training and test sets against the raw data 
    in the files specified by the user config '''
    config = configparser.ConfigParser()
    config.optionxform=str
    config_files = ['src/public_config.ini', 'src/private_config.ini', 'src/user_config.ini']
    config.read(config_files)
    # Load test data and compare against original
    raw_data_path = Path(config.get('files', 'raw_data_dir'))
    dataset_file = Path(config.get('files', 'dataset_test'))
    meta_file = Path(config.get('files', 'meta_test'))
    dataset = manager.load_dataset(dataset_file)
    meta = manager.load_meta(meta_file)
    n = meta.shape[0]
    compare_data(raw_data_path, dataset, meta, i=0)
    compare_data(raw_data_path, dataset, meta, i=n-1)
    compare_data(raw_data_path, dataset, meta)


def test_filter_data():
    ''' Test that the flattened dog behaviour database has the expected 
    number of rows. '''
    input = 'data/test_data/samson/dog_behaviour_database_samson.csv'
    target = 'data/test_data/samson/dog_behaviour_database_samson_flat.csv'
    filter_data.flatten_dog_behaviour_database(input, target)
    loaded_target = manager.load_dog_behaviour_flat_db(target)
    assert_that(loaded_target.shape[0], equal_to(6*3))
    assert_that(loaded_target.shape[1], equal_to(10))

def create_samson_dataset():
    ''' Create a dataset for samson the dog from the raw pressure sensor data csv files. '''
    source = 'data/test_data/samson/raw_data'
    dest = 'data/test_data/samson'
    num_datapoints = 100
    class_info.parse_filenames(source, dest)
    import_data.create_dataset(dest+'/good.pkl', dest+'/samson_dataset.txt', num_datapoints=num_datapoints, verbose=True)


def test_remove_samples():
    ''' From a dataset, use the dog behaviour database to identify any samples that the dog didn't 
    search (marked NS) and remove them from the dataset. Save the result as a new dataset, with
    corresponding meta data file.
    For each sample in the filtered dataset, test that it matches the raw data file that 
    it originated from. '''
    create_samson_dataset()
    database = 'data/test_data/samson/dog_behaviour_database_samson_flat.csv'
    dataset = 'data/test_data/samson/samson_dataset.txt'
    meta = 'data/test_data/samson/samson_dataset_meta.txt'
    dest = 'data/test_data/samson'
    label = 'test_filtered_dataset'
    filter_data.remove_samples(database, dataset, meta, dest, label)
    # Load dataset and test against raw data files    
    raw_data_path = Path('data/test_data/samson/raw_data')
    dataset_file = 'data/test_data/samson/'+label+'.txt'
    meta_file = 'data/test_data/samson/'+label+'_meta.txt'
    print('Testing dataset', dataset_file, 'and', meta_file)
    compare_data_files(raw_data_path, dataset_file, meta_file, i='all') 



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
    meta_np_orig = meta_np
    n = meta_np_orig.shape[0]
    dataset_np = np.array
    dataset_np = np.ones((n,20))
    for j in range(n):
        dataset_np[j] = dataset_np[j] * j

    # Shuffle, for thorough test. Then filter.
    sklearn.utils.shuffle(dataset_np, meta_np)
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
        filename_orig = meta_np_orig[int(dataset[j][0])][0]
        assert_that(filename, equal_to(filename_orig))


def test_create_balanced_dataset_from_arrays():
    ''' Test creating a balanced dataset of a given size '''
    # Create test data
    n = 40
    dataset = np.random.rand(n,5)
    meta = np.random.rand(n,3)
    for i in range(0,n):
        if i % 4 == 0:
            dataset[i][0] = 1
        else:
            dataset[i][0] = 0
        dataset[i][1] = i
        meta[i][0] = i
    
    # Get balanced dataset
    num = 20
    class_balance = 0.6
    dataset_bal, meta_bal = split_data.create_balanced_dataset_from_arrays(
        dataset, meta, num, class_balance)

    # Check output
    df = pd.DataFrame(dataset_bal)
    assert_that(df[df[0]==0].count()[0], equal_to(num*class_balance))
    assert_that(df[df[0]==1].count()[0], equal_to(num*(1-class_balance)))
    for i in range(0,num):
        assert_that(dataset_bal[i][1], equal_to(meta_bal[i][0]))

    # Test stratification
    test_size = 0.25
    dataset_train, dataset_test, __, __ = \
        train_test_split(dataset_bal, meta_bal, test_size=test_size, 
        stratify=dataset_bal[:,0])
    
    df = pd.DataFrame(dataset_train)
    assert_that(df[df[0]==0].count()[0], equal_to(num*class_balance*(1-test_size)))
    df = pd.DataFrame(dataset_test)
    assert_that(df[df[0]==0].count()[0], equal_to(num*class_balance*test_size))



def test_create_balanced_dataset():
    ''' Using samson data, test that the balanced dataset is created correctly '''
    raw_data_path = Path('data/test_data/samson/raw_data')
    dataset = 'data/test_data/samson/samson_dataset.txt'
    meta = 'data/test_data/samson/samson_dataset_meta.txt'
    num = 8
    class_balance = 0.25
    dataset = manager.load_dataset(dataset)
    meta = manager.load_meta(meta) 
    df, meta_bal = split_data.create_balanced_dataset(
        dataset, meta, num, class_balance)

    # Check output
    assert_that(df[df[0]==0].count()[0], equal_to(num*class_balance))
    assert_that(df[df[0]==1].count()[0], equal_to(num*(1-class_balance)))
    compare_data(raw_data_path, df, meta_bal, i='all') 


def test_mini_dataset():
    ''' Test that the dog-specific, balanced dataset is created correctly '''
    raw_data_path = Path('data/test_data/two_dogs/raw_data')
    dataset = 'data/test_data/two_dogs/test_filtered_dataset.txt'
    meta = 'data/test_data/two_dogs/test_filtered_dataset_meta.txt'
    dest = 'data/test_data/two_dogs'
    label = 'samson_only'
    split_data.mini_dataset(dataset, meta, 8, 0.5, 0.5, \
        dog='Samson', events_only=False, dest=dest, label=label)
    # Load and test
    dataset = 'data/test_data/two_dogs/samson_only_TRAIN.txt'
    meta = 'data/test_data/two_dogs/samson_only_TRAIN_meta.txt'
    compare_data_files(raw_data_path, dataset, meta, i='all')
    dataset = 'data/test_data/two_dogs/samson_only_TEST.txt'
    meta = 'data/test_data/two_dogs/samson_only_TEST_meta.txt'
    compare_data_files(raw_data_path, dataset, meta, i='all')



        



