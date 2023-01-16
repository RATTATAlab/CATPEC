from nltk.parse.stanford import *
from pathlib import Path
import glob
import pandas as pd
import os
import sys
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
INPUT_FILE_PREFIX = 'output_all_subset_'
INPUT_FILE_SUFFIX = '.csv'
OUTPUT_FILE_PREFIX = 'parsed_all_subset_'
OUTPUT_FILE_SUFFIX = '.csv'

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
output_dir = data_dir.joinpath('intermediate')

#-------------------------------------------
#parser用意
#-------------------------------------------
stanf_parser_path = str(tools_dir.joinpath('stanford-parser/stanford-parser-full-2020-11-17/stanford-parser.jar'))
stanf_model_path = str(tools_dir.joinpath('stanford-parser/stanford-corenlp-4.2.0-models-english.jar'))
parser = StanfordParser(path_to_jar=stanf_parser_path, path_to_models_jar=stanf_model_path)

#-------------------------------------------
#csvの読み込み
#-------------------------------------------
input_file_regex = INPUT_FILE_PREFIX + '*' + INPUT_FILE_SUFFIX
input_path_regex = str(output_dir.joinpath(input_file_regex))
input_path = select_file(input_path_regex)

if input_path == '':
    print('error :[', input_path_regex, '] is no match.')
    sys.exit()

df = pd.read_csv(input_path, dtype=str, index_col=0)
print('input file :', input_path)
print(df)

#-------------------------------------------
#convert prerequisite to tree from string
#-------------------------------------------
normalized_prereq_tree_list = []
parse_from_list = []

for i, record in df.iterrows():
    subset_class = record['SubsetClass']

    if subset_class == 'no_prerequisite':
        normalized_prereq_tree = ''
        parse_from = ''

    if subset_class == 'no_splitted':
        normalized_prereq_tree = list(parser.raw_parse(record['Prerequisite']))
        parse_from = 'prerequisite'

    if subset_class == 'text_splitted':
        normalized_prereq_tree = list(parser.raw_parse(record['Sentence']))
        parse_from = 'sentence'

    if subset_class == 'sentence_splitted':
        if pd.isna(record['Split']):
            normalized_prereq_tree = list(parser.raw_parse(record['Sentence']))
            parse_from = 'sentence'
        else:
            normalized_prereq_tree = list(parser.raw_parse(record['Split']))
            parse_from = 'split'
        
    normalized_prereq_tree_list.append(normalized_prereq_tree)
    parse_from_list.append(parse_from)

    if i % 10 == 0: 
        print('--> %05d / %05d' %(i, len(df)))
        if i != len(normalized_prereq_tree_list)-1:
            print('error, dropping record was detected.', str(i), '<->', str(len(normalized_prereq_tree_list)-1))
            break

df['ParseFrom'] = parse_from_list
df['NormalizedPrerequisite'] = normalized_prereq_tree_list
print('\n',df)

#-------------------------------------------
#CSV出力
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
