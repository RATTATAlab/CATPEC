from pathlib import Path
import glob
import pandas as pd
import os
import sys
import xml.etree.ElementTree as ET
import datetime

#-------------------------------------------
#begin
#-------------------------------------------
print('\n')
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(__file__)
print(datetime.datetime.now())
print()

#-------------------------------------------
#初期設定
#-------------------------------------------
HOME_DIR_NAME = 'CATPEC'
INPUT_FILE_PREFIX = 'classified_by_bayes_all_subset_'
INPUT_FILE_SUFFIX = '.csv'
OUTPUT_FILE_PREFIX = 'remove_garbage_all_subset_'
OUTPUT_FILE_SUFFIX = '.csv'
OUTPUT_ENCODING = 'UTF-8'

#-------------------------------------------
#関数
#-------------------------------------------
def get_home_dir(dir_name):
    path = Path(os.path.dirname(__file__))
    _index = 0
    while True:
        if path.parents[_index].name == dir_name:
            home_dir = path.parents[_index]
            break
        _index += 1
    return home_dir

def select_file(path_regex):
    input_files = glob.glob(path_regex)

    if len(input_files) == 0:
        print('error:output file can not found.')
        return ''

    elif len(input_files) == 1:
        return input_files[0]

    else:
        for _index, output_file in enumerate(input_files):
            print(_index, output_file)
        val = input('select:')
        try: int(val)
        except ValueError:
            print('error:false input.')
            sys.exit()
        else: val = int(val)
            
        if val >= 0 and val <= len(input_files) - 1:
            return input_files[val]

#-------------------------------------------
#パス設定
#-------------------------------------------
home_dir = get_home_dir(HOME_DIR_NAME)
data_dir = home_dir.joinpath('data')
tools_dir = home_dir.joinpath('tools')
input_dir = data_dir.joinpath('intermediate')
output_dir = data_dir.joinpath('intermediate')

#-------------------------------------------
#csvの読み込み
#-------------------------------------------
input_file_regex = INPUT_FILE_PREFIX + '*' + INPUT_FILE_SUFFIX
input_path_regex = str(input_dir.joinpath(input_file_regex))
input_path = select_file(input_path_regex)

if input_path == '':
    print('error :[', input_path_regex, '] is no match.')
    sys.exit()

df = pd.read_csv(input_path, dtype=str, index_col=0)
print('input file :', input_path)
print(df)

#-------------------------------------------
#remove garbage
#-------------------------------------------
cols_output = ['PatternID','Name','Prerequisite','Class']
df_output = pd.DataFrame(index=[], columns=cols_output)

for index, record in df.iterrows():
    if pd.isna(record['GarbageIs']):continue
    if record['GarbageIs'] == 'TRUE':continue

    _id = record['PatternID']
    _name = record['Name']

    if pd.isna(record['ParseFrom']):_prereq = ''
    if record['ParseFrom'] == 'prerequisite':_prereq = record['Prerequisite']
    if record['ParseFrom'] == 'sentence':_prereq = record['Sentence']
    if record['ParseFrom'] == 'split':_prereq = record['Split']

    if record['PrereqClass'] == 'unclassified':_class = record['ClassByBayes']
    else:_class = record['PrereqClass']

    _new_record = pd.Series([_id, _name, _prereq, _class], index=df_output.columns)
    df_output = pd.concat([df_output, pd.DataFrame([_new_record])], ignore_index=True)

#-------------------------------------------
#結果出力
#-------------------------------------------
now = datetime.datetime.now()
output_file_name = OUTPUT_FILE_PREFIX + now.strftime('%Y%m%d_%H%M%S' + OUTPUT_FILE_SUFFIX)
output_path = os.path.join(output_dir, output_file_name)
df_output.to_csv(output_path)

#-------------------------------------------
#end
#-------------------------------------------
print()
print(datetime.datetime.now())
print(__file__)
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('\n')
