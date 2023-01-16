from pathlib import Path
import glob
import pandas as pd
import os
import sys
import xml.etree.ElementTree as ET
import datetime
import requests

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
INPUT_FILE_PREFIX = 'remove_garbage_all_subset_'
INPUT_FILE_SUFFIX = '.csv'
OUTPUT_FILE_PREFIX = 'summaraized_all_subset_'
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
#translation by deepl
#-------------------------------------------
text_list = df['Prerequisite'].values.tolist()
text_list = ['' if pd.isna(x) else x for x in text_list]
print(text_list)

#text_list = text_list[300:310]
#print(text_list)

api_key = input('input API-KEY : ')
source_lang = 'EN'
target_lang = 'JA'

params = {
    'auth_key' : api_key,
    'text' : text_list,
    'source_lang' : source_lang,
    "target_lang": target_lang
    }

request = requests.post("https://api-free.deepl.com/v2/translate", data=params)
result = request.json()

text_list_ja = []
for text_ja in result['translations']:
    text_list_ja.append(text_ja['text'])

print(text_list_ja)

df['Prerequisite_JA'] = text_list_ja
print(df)

#-------------------------------------------
#結果出力
#-------------------------------------------
now = datetime.datetime.now()
output_file_name = OUTPUT_FILE_PREFIX + now.strftime('%Y%m%d_%H%M%S' + OUTPUT_FILE_SUFFIX)
output_path = os.path.join(output_dir, output_file_name)
df.to_csv(output_path)

#-------------------------------------------
#end
#-------------------------------------------
print()
print(datetime.datetime.now())
print(__file__)
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('\n')
