from numpy import record
from classifier import NaiveBayesClassifier as NBC
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
INPUT_FILE_PREFIX = 'classified_by_subject_all_subset_'
INPUT_FILE_SUFFIX = '.csv'
OUTPUT_FILE_PREFIX = 'classified_by_bayes_all_subset_'
OUTPUT_FILE_SUFFIX = '.csv'

#-------------------------------------------
#関数
#-------------------------------------------
def convert_tree_to_str(tokens_list):
    sent = ''
    for token in tokens_list:
        wnl = WNL()
        lower = token[0].lower()
        lemma = wnl.lemmatize(lower)
        if sent == '':
            sent = lemma
        else:
            sent = sent + ' ' + lemma
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
#split data
#-------------------------------------------
df_ad = df[df['PrereqClass'].isin(['adversary'])].copy()
df_ta = df[df['PrereqClass'].isin(['target'])].copy()
df_un = df[df['PrereqClass'].isin(['unclassified'])].copy()

#-------------------------------------------
#train bayes
#-------------------------------------------
train_ad_list = df_ad['NormalizedPrerequisite'].values.tolist()
train_ad_list = [convert_tree_to_str(delete_min_np(eval(prereq))) for prereq in train_ad_list]

train_ta_list = df_ta['NormalizedPrerequisite'].values.tolist()
train_ta_list = [convert_tree_to_str(delete_min_np(eval(prereq))) for prereq in train_ta_list]

nbc = NBC()
nbc.append_train_set('adversary', train_ad_list)
nbc.append_train_set('target', train_ta_list)
nbc.train()

print('\n--------------------------')
print('nbc.get_summary()')
print(nbc.get_summary())

#-------------------------------------------
#classify by bayes
#-------------------------------------------
class_by_bayes_list = []

wnl = WNL()
for i, _record in df.iterrows():
    if pd.isna(_record['NormalizedPrerequisite']) or pd.isna(_record['PrereqClass']):
        class_by_bayes_list.append('')
        continue

    if _record['PrereqClass'] == 'unclassified':
        prereq = _record['NormalizedPrerequisite']
        prereq = eval(prereq)
        prereq = ' '.join(prereq[0].leaves())
        prereq = prereq.lower()
        prereq = wnl.lemmatize(prereq)
        _class = nbc.classify(prereq)
    else:
        _class = ''

    class_by_bayes_list.append(_class)

df['ClassByBayes'] = class_by_bayes_list
print(df)

#-------------------------------------------
#結果出力準備
#-------------------------------------------
df_no_prrqst = df[df['SubsetClass'].isin(['no_prerequisite'])].copy()
df_no_splt = df[df['SubsetClass'].isin(['no_splitted'])].copy()
df_txt_splt = df[df['SubsetClass'].isin(['text_splitted'])].copy()
df_snt_splt = df[df['SubsetClass'].isin(['sentence_splitted'])].copy()

#-------------------------------------------
#abstract
#-------------------------------------------
print()
print('------abstract------')
print('number of normalized prerequisites :', len(df))

#-------------------------------------------
#no prerequisite
#-------------------------------------------
nof_correct_no_prrqst = len(df_no_prrqst)
nof_uncorrect_no_prrqst = 0
nof_record_no_prrqst = len(df_no_prrqst)
nof_continue_no_prrqst = 0

print()
print('------no_prerequisite------')
print('number of records   :', nof_record_no_prrqst)
print('number of correct   :', nof_correct_no_prrqst)
print('number of uncorrect :', nof_uncorrect_no_prrqst)
print('accuracy            :', nof_correct_no_prrqst/nof_record_no_prrqst)

df_grp_no_prrqst = df_no_prrqst.groupby('PatternID',sort=False).count().copy()
id_list_no_prrqst = df_grp_no_prrqst.index.tolist()
nof_pttrn_no_prrqst = len(df_grp_no_prrqst)
nof_crrct_pttrn_no_prrqst = nof_pttrn_no_prrqst
nof_uncrrct_pttrn_no_prrqst = 0
nof_impssbl_pttrn_no_prrqst = 0

print()
print('nof_pttrn             :', nof_pttrn_no_prrqst)
print('nof_clssf_pttrn       :', nof_crrct_pttrn_no_prrqst)
print('nof_unclssf_pttrn     :', nof_uncrrct_pttrn_no_prrqst)
print('nof_impssbl_pttrn     :', nof_impssbl_pttrn_no_prrqst)

#-------------------------------------------
#no splitted
#-------------------------------------------
nof_correct_subject_no_splt = 0
nof_uncorrect_subject_no_splt = 0
nof_correct_bayes_no_splt = 0
nof_uncorrect_bayes_no_splt = 0
nof_record_no_splt = 0
nof_continue_no_splt = 0
evaluation = ''
evaluation_list = []

for i, _record in df_no_splt.iterrows():
    nof_record_no_splt += 1

    if _record['PrereqLabel'] == '-':
        nof_continue_no_splt += 1
        evaluation = 'impossible'
        evaluation_list.append(evaluation)
        continue

    if _record['PrereqClass'] == 'unclassified':
        if _record['ClassByBayes'] == _record['PrereqLabel']:
            nof_correct_bayes_no_splt += 1
            evaluation = 'correct'
        else:
            nof_uncorrect_bayes_no_splt += 1
            evaluation = 'uncorrect'
    else:
        if _record['PrereqClass'] == _record['PrereqLabel']:
            nof_correct_subject_no_splt += 1
            evaluation = 'correct'
        else:
            nof_uncorrect_subject_no_splt +=1
            evaluation = 'uncorrect'

    evaluation_list.append(evaluation)

print()
print('------no_splitted------')
print('number of records      :',nof_record_no_splt)
print('nof_continue           :',nof_continue_no_splt)
print('nof_correct_subject    :',nof_correct_subject_no_splt)
print('nof_uncorrect_subject  :',nof_uncorrect_subject_no_splt)
print('nof_correct_bayes      :',nof_correct_bayes_no_splt)
print('nof_uncorrect_bayes    :',nof_uncorrect_bayes_no_splt)
print('accuracy               :',(nof_correct_subject_no_splt + nof_correct_bayes_no_splt)/(nof_record_no_splt - nof_continue_no_splt))

df_no_splt['Evaluation'] = evaluation_list

df_grp_no_splt = df_no_splt.groupby('PatternID',sort=False).count().copy()
id_list_no_splt = df_grp_no_splt.index.tolist()
nof_pttrn_no_splt = len(df_grp_no_splt)
nof_crrct_pttrn_no_splt = 0
nof_uncrrct_pttrn_no_splt = 0
nof_impssbl_pttrn_no_splt = 0

for id in id_list_no_splt:
    if df_no_splt[df_no_splt['PatternID'].isin([id])]['Evaluation'].isin(['impossible']).any():nof_impssbl_pttrn_no_splt += 1
    elif df_no_splt[df_no_splt['PatternID'].isin([id])]['Evaluation'].isin(['uncorrect']).any():nof_uncrrct_pttrn_no_splt += 1
    else:nof_crrct_pttrn_no_splt += 1

print()
print('nof_pttrn             :', nof_pttrn_no_splt)
print('nof_clssf_pttrn       :', nof_crrct_pttrn_no_splt)
print('nof_unclssf_pttrn     :', nof_uncrrct_pttrn_no_splt)
print('nof_impssbl_pttrn     :', nof_impssbl_pttrn_no_splt)

#-------------------------------------------
#text splitted
#-------------------------------------------
print()
print('------text_splitted------')
nof_correct_subject_txt_splt = 0
nof_uncorrect_subject_txt_splt = 0
nof_correct_bayes_txt_splt = 0
nof_uncorrect_bayes_txt_splt = 0
nof_record_txt_splt = 0
nof_continue_txt_splt = 0
evaluation = ''
evaluation_list = []

for i, _record in df_txt_splt.iterrows():
    nof_record_txt_splt += 1

    if _record['PrereqLabel'] == '-':
        nof_continue_txt_splt += 1
        evaluation = 'impossible'
        evaluation_list.append(evaluation)
        continue

    if _record['PrereqClass'] == 'unclassified':
        if _record['ClassByBayes'] == _record['PrereqLabel']:
            nof_correct_bayes_txt_splt += 1
            evaluation = 'correct'
        else:
            nof_uncorrect_bayes_txt_splt += 1
            evaluation = 'uncorrect'
    else:
        if _record['PrereqClass'] == _record['PrereqLabel']:
            nof_correct_subject_txt_splt += 1
            evaluation = 'correct'
        else:
            nof_uncorrect_subject_txt_splt +=1
            evaluation = 'uncorrect'

    evaluation_list.append(evaluation)

print('number of records      :',nof_record_txt_splt)
print('nof_continue           :',nof_continue_txt_splt)
print('nof_correct_subject    :',nof_correct_subject_txt_splt)
print('nof_uncorrect_subject  :',nof_uncorrect_subject_txt_splt)
print('nof_correct_bayes      :',nof_correct_bayes_txt_splt)
print('nof_uncorrect_bayes    :',nof_uncorrect_bayes_txt_splt)
print('accuracy               :',(nof_correct_subject_txt_splt + nof_correct_bayes_txt_splt)/(nof_record_txt_splt - nof_continue_txt_splt))

df_txt_splt['Evaluation'] = evaluation_list

df_grp_txt_splt = df_txt_splt.groupby('PatternID',sort=False).count().copy()
id_list_txt_splt = df_grp_txt_splt.index.tolist()
nof_pttrn_txt_splt = len(df_grp_txt_splt)
nof_crrct_pttrn_txt_splt = 0
nof_uncrrct_pttrn_txt_splt = 0
nof_impssbl_pttrn_txt_splt = 0

for id in id_list_txt_splt:
    if df_txt_splt[df_txt_splt['PatternID'].isin([id])]['Evaluation'].isin(['impossible']).any():nof_impssbl_pttrn_txt_splt += 1
    elif df_txt_splt[df_txt_splt['PatternID'].isin([id])]['Evaluation'].isin(['uncorrect']).any():nof_uncrrct_pttrn_txt_splt += 1
    else:nof_crrct_pttrn_txt_splt += 1

print()
print('nof_pttrn    :', nof_pttrn_txt_splt)
print('nof_clssf_pttrn :', nof_crrct_pttrn_txt_splt)
print('nof_unclssf_pttrn :', nof_uncrrct_pttrn_txt_splt)
print('nof_impssbl_pttrn :', nof_impssbl_pttrn_txt_splt)

#-------------------------------------------
#sentence splitted
#-------------------------------------------
print()
print('------sentence_splitted------')
nof_correct_subject_snt_splt = 0
nof_uncorrect_subject_snt_splt = 0
nof_correct_bayes_snt_splt = 0
nof_uncorrect_bayes_snt_splt = 0
nof_record_snt_splt = 0
nof_continue_snt_splt = 0
evaluation = ''
evaluation_list = []

for i, _record in df_snt_splt.iterrows():
    if _record['GarbageIs'] == 'TRUE':
        evaluation = ''
        evaluation_list.append(evaluation)
        continue

    nof_record_snt_splt += 1

    if _record['PrereqLabel'] == '-':
        nof_continue_snt_splt += 1
        evaluation = 'impossible'
        evaluation_list.append(evaluation)
        continue

    if _record['PrereqClass'] == 'unclassified':
        if _record['ClassByBayes'] == _record['PrereqLabel']:
            nof_correct_bayes_snt_splt += 1
            evaluation = 'correct'
        else:
            nof_uncorrect_bayes_snt_splt += 1
            evaluation = 'uncorrect'
    else:
        if _record['PrereqClass'] == _record['PrereqLabel']:
            nof_correct_subject_snt_splt += 1
            evaluation = 'correct'
        else:
            nof_uncorrect_subject_snt_splt +=1
            evaluation = 'uncorrect'

    evaluation_list.append(evaluation)

print('number of records      :',nof_record_snt_splt)
print('nof_continue           :',nof_continue_snt_splt)
print('nof_correct_subject    :',nof_correct_subject_snt_splt)
print('nof_uncorrect_subject  :',nof_uncorrect_subject_snt_splt)
print('nof_correct_bayes      :',nof_correct_bayes_snt_splt)
print('nof_uncorrect_bayes    :',nof_uncorrect_bayes_snt_splt)
print('accuracy               :',(nof_correct_subject_snt_splt + nof_correct_bayes_snt_splt)/(nof_record_snt_splt - nof_continue_snt_splt))

df_snt_splt['Evaluation'] = evaluation_list

df_grp_snt_splt = df_snt_splt.groupby('PatternID',sort=False).count().copy()
id_list_snt_splt = df_grp_snt_splt.index.tolist()
nof_pttrn_snt_splt = len(df_grp_snt_splt)
nof_crrct_pttrn_snt_splt = 0
nof_uncrrct_pttrn_snt_splt = 0
nof_impssbl_pttrn_snt_splt = 0

for id in id_list_snt_splt:
    if df_snt_splt[df_snt_splt['PatternID'].isin([id])]['Evaluation'].isin(['impossible']).any():nof_impssbl_pttrn_snt_splt += 1
    elif df_snt_splt[df_snt_splt['PatternID'].isin([id])]['Evaluation'].isin(['uncorrect']).any():nof_uncrrct_pttrn_snt_splt += 1
    else:nof_crrct_pttrn_snt_splt += 1

print()
print('nof_pttrn    :', nof_pttrn_snt_splt)
print('nof_clssf_pttrn :', nof_crrct_pttrn_snt_splt)
print('nof_unclssf_pttrn :', nof_uncrrct_pttrn_snt_splt)
print('nof_imppsbl_pttrn :', nof_impssbl_pttrn_snt_splt)

#-------------------------------------------
#summary
#-------------------------------------------
nof_all_record = (nof_record_no_prrqst + nof_record_no_splt + nof_record_txt_splt + nof_record_snt_splt)

nof_all_correct = (
    nof_correct_no_prrqst +
    nof_correct_subject_no_splt + nof_correct_bayes_no_splt +
    nof_correct_subject_txt_splt + nof_correct_bayes_txt_splt +
    nof_correct_subject_snt_splt + nof_correct_bayes_snt_splt
    )

nof_all_uncorrect = (
    nof_uncorrect_no_prrqst +
    nof_uncorrect_subject_no_splt + nof_uncorrect_bayes_no_splt +
    nof_uncorrect_subject_txt_splt + nof_uncorrect_bayes_txt_splt +
    nof_uncorrect_subject_snt_splt + nof_uncorrect_bayes_snt_splt
    )

nof_all_continue = (nof_continue_no_prrqst + nof_continue_no_splt + nof_continue_txt_splt + nof_continue_snt_splt)

continue_ratio = (nof_all_continue/nof_all_record)
accuracy = (nof_all_correct/(nof_all_record - nof_all_continue))

print()
print('------summary------')
print('nof_all_record         :',nof_all_record)
print('nof_all_correct        :',nof_all_correct)
print('nof_all_uncorrect      :',nof_all_uncorrect)
print('nof_all_continue       :',nof_all_continue)
print('accuracy               :',accuracy)
print('continue ratio         :',continue_ratio)

nof_all_pttrn = (nof_pttrn_no_prrqst + nof_pttrn_no_splt + nof_pttrn_txt_splt + nof_pttrn_snt_splt)
nof_all_crrct_pttrn = (nof_crrct_pttrn_no_prrqst + nof_crrct_pttrn_no_splt + nof_crrct_pttrn_txt_splt + nof_crrct_pttrn_snt_splt)
nof_all_uncrrct_pttrn = (nof_uncrrct_pttrn_no_prrqst + nof_uncrrct_pttrn_no_splt + nof_uncrrct_pttrn_txt_splt + nof_uncrrct_pttrn_snt_splt)
nof_all_impssbl_pttrn = (nof_impssbl_pttrn_no_prrqst + nof_impssbl_pttrn_no_splt + nof_impssbl_pttrn_txt_splt + nof_impssbl_pttrn_snt_splt)
crrct_pttrn_ratio = nof_all_crrct_pttrn / nof_all_pttrn

print()
print('nof_all_pttrn          :',nof_all_pttrn)
print('nof_all_crrct_pttrn    :',nof_all_crrct_pttrn)
print('nof_all_uncrrct_pttrn  :',nof_all_uncrrct_pttrn)
print('nof_all_impssbl_pttrn  :',nof_all_impssbl_pttrn)
print('correct ratio          :',crrct_pttrn_ratio)

#-------------------------------------------
#csv
#-------------------------------------------
df_output = pd.concat([df_no_prrqst, df_no_splt, df_txt_splt, df_snt_splt], ignore_index=True)
print('\n',df_output)

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
