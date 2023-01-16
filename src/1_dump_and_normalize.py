from nltk import tokenize
from nltk.parse.stanford import *
from nltk.tree import *
from pathlib import Path
import datetime
import math
import os
import pandas as pd
import time
import xml.etree.ElementTree as ET



#時間計測
start_time = time.time()

#パス設定
home_dir_name = 'CATPEC'

path = Path(os.path.dirname(__file__))
_index = 0
while True:
    if path.parents[_index].name == home_dir_name:
        home_dir = path.parents[_index]
        break
    _index += 1

data_dir = home_dir.joinpath('data')
tools_dir = home_dir.joinpath('tools')
output_dir = data_dir.joinpath('intermediate')

capec_path = data_dir.joinpath('capec/CAPEC_3.8_MechanismsOfAttack.xml')
stanf_parser_path = str(tools_dir.joinpath('stanford-parser/stanford-parser-full-2020-11-17/stanford-parser.jar'))
stanf_model_path = str(tools_dir.joinpath('stanford-parser/stanford-corenlp-4.2.0-models-english.jar'))
parser = StanfordParser(path_to_jar=stanf_parser_path, path_to_models_jar=stanf_model_path)

#dataframeの用意
cols_raw_capec = ['PatternID','PrereqID','Name','Prerequisite']
df_raw_capec = pd.DataFrame(index=[], columns=cols_raw_capec)

cols_sent_tokenized = ['PatternID','PrereqID','SentID','Name','Prerequisite','Sentence']
df_sent_tokenized = pd.DataFrame(index=[], columns=cols_sent_tokenized)

cols_output = ['PatternID','PrereqID','SentID','SplitID','Name','Prerequisite','SentLabel','Sentence','SplitLabel','Split']
df_output = pd.DataFrame(index=[], columns=cols_output)



#CAPECからの読み込み
capec_tree = ET.parse(capec_path)
capec_root = capec_tree.getroot()

for _sub1 in capec_root.iter('{http://capec.mitre.org/capec-3}Attack_Patterns'):
    for _sub2 in _sub1.iter('{http://capec.mitre.org/capec-3}Attack_Pattern'):
        _pattern_id = str(_sub2.attrib['ID']).zfill(4)
        _pattern_name = _sub2.attrib['Name']

        #Prerequisitesを含む場合とそうでない場合
        if not _sub2.findall('{http://capec.mitre.org/capec-3}Prerequisites'):
            _prerequsite = ''
            _prereq_id = ''
            _record = pd.Series([_pattern_id,_prereq_id,_pattern_name,_prerequsite], index=df_raw_capec.columns)
            df_raw_capec = pd.concat([df_raw_capec, pd.DataFrame([_record])], ignore_index=True)
        else:
            for _sub3 in _sub2.iter('{http://capec.mitre.org/capec-3}Prerequisites'): 
                for _index, _sub4 in enumerate(_sub3.findall('{http://capec.mitre.org/capec-3}Prerequisite')):
                    _prerequsite = _sub4.text
                    _prereq_id = _pattern_id + '-' + str(_index).zfill(2)
                    _record = pd.Series([_pattern_id,_prereq_id,_pattern_name,_prerequsite], index=df_raw_capec.columns)
                    df_raw_capec = pd.concat([df_raw_capec, pd.DataFrame([_record])], ignore_index=True)

print('\n',df_raw_capec)



#Sentence単位に分割
for _index, _record in df_raw_capec.iterrows():

    #Prerequisitesを含まない場合
    if _record['Prerequisite'] == '':
        _sentence_id = ''
        _sentence = ''
        _new_record = pd.Series([
            str(_record['PatternID']),_record['PrereqID'],_sentence_id, _record['Name'],_record['Prerequisite'],_sentence
            ], index=df_sent_tokenized.columns)

        df_sent_tokenized = pd.concat([df_sent_tokenized, pd.DataFrame([_new_record])], ignore_index=True)

    #Prerequisitesを含む場合
    if _record['Prerequisite'] != '':
        _sentences_list = tokenize.sent_tokenize(_record['Prerequisite'])

        for _sent_index, _sentence in enumerate(_sentences_list):
            _sentence_id = _record['PrereqID'] + '-' +  str(_sent_index).zfill(2)
            _new_record = pd.Series([
                str(_record['PatternID']),_record['PrereqID'],_sentence_id, _record['Name'],_record['Prerequisite'],_sentence
                ], index=df_sent_tokenized.columns)

            df_sent_tokenized = pd.concat([df_sent_tokenized, pd.DataFrame([_new_record])], ignore_index=True)

print('\n',df_sent_tokenized)



#Sentenceを等位接続詞でさらに分割
_lower_index = 0
_upper_index = len(df_sent_tokenized)
_range = _upper_index - _lower_index
_index_in_range = 0
_threshold = 1

for _index1, _record in df_sent_tokenized.iterrows():

    if _index1 <= _lower_index:continue
    if _index1 >= _upper_index:break

    _index_in_range += 1
    _percentage = (_index_in_range / _range) * 100

    if _percentage >= _threshold:
        print(math.floor(_percentage), '%', 'runtime =', math.floor(time.time() - start_time), '[s]')
        _threshold = math.floor(_percentage + 1)

    #Sentenceが空の場合とそうでない場合
    if _record['Sentence'] == '':
        _split_id = ''
        _sentence_label = ''
        _split_label = ''
        _split = ''
        _new_record = pd.Series([
            _record['PatternID'],_record['PrereqID'],_record['SentID'],_split_id,
            _record['Name'],_record['Prerequisite'],_sentence_label,_record['Sentence'],_split_label,_split
            ], index=df_output.columns)

        df_output = pd.concat([df_output, pd.DataFrame([_new_record])], ignore_index=True)
    else:
        _consituency_tree = list(parser.raw_parse(_record['Sentence']))
        for _tree_root in _consituency_tree:
            for _tree_level_1 in _tree_root:
                _number_of_cc = 0

                #CCをカウント
                for _tree_level_2 in _tree_level_1:
                    if _tree_level_2.label() == 'CC':
                        _number_of_cc += 1

                #CCを含む場合とそうでない場合
                if _number_of_cc >= 1:
                    for _index2, _tree_level_2 in enumerate(_tree_level_1):
                        _split_id = _record['SentID'] + '-' + str(_index2).zfill(2)
                        _sentence_label = _tree_level_1.label()
                        _split_label = _tree_level_2.label()
                        _split = ' '.join(_tree_level_2.leaves())

                        _new_record = pd.Series([
                            _record['PatternID'],_record['PrereqID'],_record['SentID'],_split_id,
                            _record['Name'],_record['Prerequisite'],_sentence_label,_record['Sentence'],_split_label,_split
                            ], index=df_output.columns)

                        df_output = pd.concat([df_output, pd.DataFrame([_new_record])], ignore_index=True)
                else:
                        _split_id = ''
                        _sentence_label = _tree_level_1.label()
                        _split_label = ''
                        _split = ''
                        _new_record = pd.Series([
                            _record['PatternID'],_record['PrereqID'],_record['SentID'],_split_id,
                            _record['Name'],_record['Prerequisite'],_sentence_label,_record['Sentence'],_split_label,_split
                            ], index=df_output.columns)

                        df_output = pd.concat([df_output, pd.DataFrame([_new_record])], ignore_index=True)

print('\n',df_output)
now = datetime.datetime.now()
output_file_name = 'output_from_capec_' + now.strftime('%Y%m%d_%H%M%S' + '.csv')
output_path = str(output_dir.joinpath(output_file_name))

df_output.to_csv(output_path)

#実行時間計測
stop_time = time.time()
run_time = math.floor(stop_time - start_time)
print('runtime =', run_time, '[s]')
