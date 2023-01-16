from nltk.parse.stanford import *
from nltk.stem.wordnet import WordNetLemmatizer as WNL
from pathlib import Path
import glob
import pandas as pd
import os
import sys
import nltk
import xml.etree.ElementTree as ET
import datetime
import re

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
INPUT_FILE_PREFIX = 'selected_parsed_all_subset_'
INPUT_FILE_SUFFIX = '.csv'
OUTPUT_FILE_PREFIX = 'classified_by_subject_all_subset_'
OUTPUT_FILE_SUFFIX = '.csv'

#-------------------------------------------
#関数
#-------------------------------------------
def remove_stop_words(tokens_list):
    sent = ''
    for token in tokens_list:
        if (
            re.findall('NN.?',token[1]) or #名詞
            re.findall('VB.?',token[1]) or #動詞
            re.findall('JJ.?',token[1]) or #形容詞
            re.findall('RB.?',token[1])    #副詞
            ):
            if sent == '':
                sent = token[0]
            else:
                sent = sent + ' ' + token[0]
    return sent

def delete_min_np(consituency_tree):
    _sentence = []
    if len(consituency_tree) >= 1:
        for _sub_tree in consituency_tree:
            _phrase = _sub_tree.pos()

            if _sub_tree.label() == 'ROOT' or _sub_tree.label() == 'S':
                _phrase = delete_min_np(_sub_tree)

            if _sub_tree.label() == 'NP':
                _phrase = delete_min_np(_sub_tree)

                if _phrase == _sub_tree.pos():
                    _phrase = []

            _sentence = _sentence + _phrase
    return _sentence

def search_min_np(consituency_tree):
    if len(consituency_tree) >= 1:
        for _sub_tree in consituency_tree:

            if _sub_tree.label() == 'ROOT' or _sub_tree.label() == 'S':
                _subject = search_min_np(_sub_tree)

                if _subject == '':
                    return ''
                else:
                    return _subject

            if _sub_tree.label() == 'NP':
                _subject = search_min_np(_sub_tree)

                if _subject == '':
                    return ' '.join(_sub_tree.leaves())
                else:
                    return _subject

    return ''

def classifer_bysubject(subject):
    if subject == '': return 'unclassified'

    adversaries_list = ['adversary','attacker']
    targets_list = ['system','application','target','victim','host','software','server']

    tokens = nltk.word_tokenize(subject)
    wnl = WNL()
    adversary_is = False
    target_is = False

    for token in tokens:
        lower = token.lower()
        lemma = wnl.lemmatize(lower)

        if lemma in adversaries_list: adversary_is = True
        if lemma in targets_list: target_is = True

    if adversary_is == True and target_is == False: return 'adversary'
    elif adversary_is == False and target_is == True: return 'target'
    else: return 'unclassified'

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
#classify by subject
#-------------------------------------------
parsed_prereq_list = df['NormalizedPrerequisite'].values.tolist()
garbage_is_list = df['GarbageIs'].values.tolist()
class_list = []

for i, parsed_prereq in enumerate(parsed_prereq_list):
    if pd.isna(parsed_prereq):
        _class = ''
    else:
        prereq_tree = eval(parsed_prereq)
        subject = search_min_np(prereq_tree)
        _class = classifer_bysubject(subject)

    if garbage_is_list[i] == 'True':
        _class = ''

    class_list.append(_class)

df['PrereqClass'] = class_list
df['PrereqLabel'] = ''
print('\n',df)

#-------------------------------------------
#CSV出力
#-------------------------------------------
now = datetime.datetime.now()
output_file_name = OUTPUT_FILE_PREFIX + now.strftime('%Y%m%d_%H%M%S' + OUTPUT_FILE_SUFFIX)
output_path = os.path.join(output_dir, output_file_name)
df.to_csv(output_path)

#-------------------------------------------
#結果出力
#-------------------------------------------
nof_prereq = len(df)
nof_ad_prereq = len(df[df['PrereqClass'].isin(['adversary'])])
nof_ta_prereq = len(df[df['PrereqClass'].isin(['target'])])
nof_un_prereq = len(df[df['PrereqClass'].isin(['unclassified'])])

df_grp = df.groupby('PatternID',sort=False).count().copy()
id_list = df_grp.index.tolist()
nof_pttrn = len(df_grp)
nof_clssf_pttrn = 0
nof_unclssf_pttrn = 0
for id in id_list:
    if df[df['PatternID'].isin([id])]['PrereqClass'].isin(['unclassified']).any():
        nof_unclssf_pttrn += 1
    else:
        nof_clssf_pttrn += 1

print()
print('nof_prereq    :', nof_prereq)
print('nof_ad_prereq :', nof_ad_prereq)
print('nof_ta_prereq :', nof_ta_prereq)
print('nof_un_prereq :', nof_un_prereq)

print()
print('nof_pttrn    :', nof_pttrn)
print('nof_clssf_pttrn :', nof_clssf_pttrn)
print('nof_unclssf_pttrn :', nof_unclssf_pttrn)

#-------------------------------------------
#end
#-------------------------------------------
print()
print(datetime.datetime.now())
print(__file__)
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('\n')
