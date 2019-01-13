#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
from pathlib import Path
import argparse
import configparser

import pandas as pd

# Load config
config = configparser.ConfigParser()
config.optionxform=str
config_files = ['src/public_config.ini', 'src/private_config.ini']
config.read(config_files)

dog_names = config._sections['dog_names']
positions = config._sections['positions']
exclude_dirs = config._sections['exclude_dirs']
exclude_file_text = config._sections['exclude_file_text']
alt_format1_dirs = config._sections['alt_format1_dirs']


def dog_name(file_name):
    ''' Return dog name or empty string if not found. '''
    this_dog = ''
    for name in dog_names:
        if file_name.find(name) >= 0:
            this_dog = dog_names[name]
            break
    return this_dog


def position(file_name):
    ''' Return the position of the positive sample or empty string if not found. '''
    this_position = ''
    for p in positions:
        if file_name.find(p) >= 0:
            this_position = positions[p]
            break
    return this_position


def exclude_dir(file):
    ''' Return true if this file should be excluded based on its directory. '''   
    # Is this directory in the exclusion list?
    for d in exclude_dirs:
        if file.match('*/'+d+'/*') or file.match('*/'+d+'/**/*'):
            return True
    return False


def alternative_format_dir(file):
    ''' Return true if this file is in a dir that holds files
    with names in an alternative format. '''   
    # Is this directory in the list?
    for d in alt_format1_dirs:
        if d in file.parents._parts:
            return True
    return False


def exclude_file(file):  
    ''' Return true if this file should be excluded. '''             
    # Does the file name include text that is in the exclusion list?
    for t in exclude_file_text:
        if file.name.find(t) >= 0:
            return True   
    return False


def last_three_underscores(file_name):
    ''' Return the positions of the last three underscores in the file name. '''
    do_print = False
    if do_print: print(file_name)
    n = len(file_name)
    last = file_name.rfind('_',0,n)
    if do_print: print('last',last)
    last_m1 = file_name.rfind('_',0,last)
    if do_print: print('last_m1',last_m1)
    last_m2 = file_name.rfind('_',0,last_m1)
    if do_print: print('last_m2',last_m2)
    return last_m2, last_m1, last


def run_and_pass_no(file_name):
    ''' Return the run number and pass number from the file name or return empty if not found.'''
    last_m2, last_m1, last = last_three_underscores(file_name)
    pass_no = file_name[last_m1+1:last]
    if not pass_no.isdigit():
        print(file_name, 'Pass number not found. Will try to handle any trailing underscore.')
        # Handle files with trailing underscore, e.g. 2018_07_24-11_51-Rex_10_1_T2_.csv
        if file_name[last] == '_':
            file_name = file_name[:last]
            last_m2, last_m1, last = last_three_underscores(file_name)
            pass_no = file_name[last_m1+1:last]
    pass_no = int(pass_no)

    run_no = file_name[last_m2+1:last_m1]
    if not run_no.isdigit():
        print(file_name)
    run_no = int(run_no)

    if pass_no > 4 or pass_no < 1 or run_no < 1: 
        print('############ Pass no is too high or negative run or pass no found at ', file_name, 'pass no:', pass_no, 'run no:', run_no, '######################################')
        run_no = ''
        pass_no = ''
    return run_no, pass_no


def timestamp(file_name):
    ''' Return timestamp from the file_name. '''
    time_stamp = file_name[:16]
    time_stamp = datetime.datetime.strptime(time_stamp, '%Y_%m_%d-%H_%M')
    return time_stamp


def extract_info(file, info):
    ''' Extract class info from file name and populate the info
    object. Return False if the expected info can't be extracted. '''
    file_name = file.name
    # Get the dog's name from the file name.
    this_dog = dog_name(file_name)
    if not this_dog:
        #print('Dog name not found for file', file_name)
        info.skipped.append((file, 'dog name not found'))   
        return False

    this_position = position(file_name)
    if not this_position:
        print('Position not found for file', file_name)
        info.skipped.append((file, 'position of positive sample not found'))
        return False

    run_no, pass_no = run_and_pass_no(file_name)
    if not run_no or not pass_no:
        #print('Run or pass number not found for file', file_name)
        info.skipped.append((file, 'run or pass number not found'))
        return False

    time_stamp = timestamp(file_name)
    info.good.append((file, time_stamp, this_dog, run_no, pass_no, this_position))
    return True

def alt_extract_info(file, info):
    ''' Extract class info from a file name in the alternative format1.
    Populate the info object. Return False if the expected info can't 
    be extracted. '''
    file_name = file.name
    timestamp, dog, run_no, pass_no, position = \
        alternative_format(file_name)

    if not timestamp:
        info.skipped.append((file, 'time stamp could not be extracted (alt format 1 expected)'))   
        return False

    if not dog:
        info.skipped.append((file, 'dog name not found'))   
        return False

    if not position:
        print('Position not found for file', file_name)
        info.skipped.append((file, 'position of positive sample not found'))
        return False

    run_no, pass_no = run_and_pass_no(file_name)
    if not run_no or not pass_no:
        info.skipped.append((file, 'run or pass number not found'))
        return False

    info.good.append((file, timestamp, dog, run_no, pass_no, position))
    return True

def alternative_format(file_name):
    ''' Get class info from a file name that is this alternative format.
    '''
    # 10-07-2017_Rex_2_2_1-123ppm
    # 10-07-2017_Rex_10_1_2-1.1m-newoil123
    # 10-07-2017_Spike_5_1_0

    time_stamp = dog = run_no = pass_no = position =''
    num_underscores = file_name.count('_')
    if num_underscores != 4:
        return time_stamp, dog, run_no, pass_no, position

    time_stamp = file_name[:10]
    time_stamp = datetime.datetime.strptime(time_stamp, '%d-%m-%Y')

    last_m2, __, last = last_three_underscores(file_name)
    name_str = file_name[11:last_m2]
    dog = dog_name(name_str)

    run_no, pass_no = run_and_pass_no(file_name)

    pos = file_name[last+1]
    position = ''
    if pos == '1':
        position = 'T1'
    elif pos == '2':
        position = 'T2'
    elif pos == '3':
        position = 'T3'
    elif pos == '0':
        position = 'B'

    return time_stamp, dog, run_no, pass_no, position

class FileInfo:
    ''' Store information extracted from file names. '''
    def __init__(self):
        self.good = [] # Good files and the class info extracted from them.
        self.skipped = [] # The files that were skipped and the reason for skipping them.


def parse_filenames(source, dest=''):
    ''' Get class info from file names in the source dir and save it
    in the dest dir as good.pkl and skipped.pkl. Print this "good" 
    info and print a list of "skipped" files where the info could 
    not be found. Return good and skipped info as Pandas 
    dataframes. '''

    info = FileInfo()
    files = Path(source).rglob('*.csv')
    for file in files:       
        # Exclude certain directories and files containing certain text.
        if exclude_dir(file):
            info.skipped.append((file,'directory excluded'))   
            continue
        if exclude_file(file):
            info.skipped.append((file,'file excluded'))   
            continue

        if not alternative_format_dir(file):
            extract_info(file, info)
        else:
            alt_extract_info(file, info)

    # Print out what was found.
    print('done')
    good_files = pd.DataFrame(info.good, columns=['file', 'timestamp', 'dog', 'run', 'pass', 'position'])
    skipped_files = pd.DataFrame(info.skipped, columns=['file', 'reason'])
    print('number of good_files', good_files.count())
    print('number of skipped_files', skipped_files.count())
    print('The reasons for excluding files are', skipped_files.reason.unique())
    
    print('The files where reason is - position of positive sample not found')
    df = skipped_files['file'][skipped_files['reason']=='position of positive sample not found']
    for r in df:
        print(r)

    print('The files where reason is - run or pass number not found')
    df = skipped_files['file'][skipped_files['reason']=='run or pass number not found']
    for r in df:
        print(r)

    print('The files where reason is - file excluded')
    df = skipped_files['file'][skipped_files['reason']=='file excluded']
    for r in df:
        print(r)

    # Save to file
    if dest:
        save_good = Path(dest+'/good.pkl')
        save_skipped = Path(dest+'/skipped.pkl')
        print('Saving data to', save_good, 'and', save_skipped)
        good_files.to_pickle(save_good)
        skipped_files.to_pickle(save_skipped)

    print('number of good_files', good_files.count())
    print('number of skipped_files', skipped_files.count())
    return good_files, skipped_files


def main():
    parser = argparse.ArgumentParser(description='Get class infomation from file names in source directory.')
    parser.add_argument('source', help='source directory')
    parser.add_argument('--dest', default='',
        help='destination directory for saving the class information files')
    args = parser.parse_args()
    parse_filenames(args.source, args.dest)

if __name__ == "__main__":
    main()



