import datetime
import pandas as pd

from data_processing import class_info

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

def test_class_info():
    source = 'notebooks/test_data/raw_data'
    good, skipped = class_info.class_info(source)
    assert(good.shape == (3,6))
    assert(skipped.shape == (0,2))