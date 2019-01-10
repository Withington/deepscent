from pathlib import Path
import datetime
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
    path = Path('do/alt_format/10-07-2017_Spike_5_1_0.csv')
    assert(class_info.alternative_format_dir(path))

def test_class_info():
    source = 'data/test_data/raw_data'
    good, skipped = class_info.class_info(source)
    assert_that(good.shape, equal_to((8,6)))
    assert_that(skipped.shape, equal_to((6,2)))

    file0 = good.at[0,'file']
    assert_that(file0.name, equal_to('2017_11_06-13_52-Rex_1_1_T1_.csv'))
    assert(good.at[0,'dog'] == 'Rex')
    assert(good.at[0,'run'] == 1)
    assert(good.at[0,'pass'] == 1)
    assert(good.at[0,'position'] == 'T1')
    time = good.at[0,'timestamp']
    expected = datetime.datetime(2017, 11, 6, 13, 52)
    assert(time == expected)


def test_raw_data_size():
    input = 'data/test_data/class_info/good.pkl'
    max = import_data.data_size(input)
    assert(max == 12000)

def test_import_data():
    input = 'data/test_data/class_info/good.pkl'
    target = ''
    cols = 6000
    dataset_shape = import_data.import_data(input, target, cols)
    assert(dataset_shape == (4*3,cols+1))