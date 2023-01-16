from pathlib import Path
import glob
import pandas as pd
import os
import sys
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
INPUT_FILE_PREFIX = 'output_from_capec_'
INPUT_FILE_SUFFIX = '.csv'
OUTPUT_NOPRRQ_FILE_PREFIX = 'subset_no_prereq_'
OUTPUT_NOSPLT_FILE_PREFIX = 'subset_no_splitted_'
OUTPUT_TXSPLT_FILE_PREFIX = 'subset_text_splitted_'
OUTPUT_STSPLT_FILE_PREFIX = 'subset_sent_splitted_'
OUTPUT_ALL_FILE_PREFIX = 'output_all_subset_'
OUTPUT_FILE_SUFFIX = '.csv'

#-------------------------------------------
#関数
#-------------------------------------------
def pattern_classifier(target_id, df):
    sent_splitted_is, text_splitted_is = False, False

    df_tmp = df[df['PatternID'].isin([target_id])].copy()

    if df_tmp['SentID'].str.endswith('01').any(): text_splitted_is = True
    if not df_tmp['SplitID'].isnull().all(): sent_splitted_is = True

    if df_tmp['PrereqID'].isnull().all(): return 'no_prereq'
    if text_splitted_is == False and sent_splitted_is == False: return 'raw'
    if text_splitted_is == True and sent_splitted_is == False: return 'text_splitted'
    if sent_splitted_is == True: return 'sent_splitted'

def pattern_classifier_demo(target_id, df):
    sent_splitted_is, text_splitted_is = False, False

    #idが一致する行だけ抽出
    df_tmp = df[df['PatternID'].isin([target_id])].copy()

    print()
    print('--------------------------------')
    print('id :', target_id)
    print(df_tmp)

    #sentIDが一つでも'-01'だったら分割が起きている
    if df_tmp['SentID'].str.endswith('01').any():
        text_splitted_is = True
        print('text_splitted_is', text_splitted_is)

    #splitIDが全てNaNで無ければ分割が起きている
    if not df_tmp['SplitID'].isnull().all():
        sent_splitted_is = True
        print('sent_splitted_is', sent_splitted_is)

    #prerequisiteが無い
    if df_tmp['PrereqID'].isnull().all():
        print('class is', 'no_prereq')
        return 'no_prereq'

    #何も起きていない
    if text_splitted_is == False and sent_splitted_is == False:
        print('class is', 'raw')
        return 'raw'

    #sentenceの分割は無いがtextの分割はある
    if text_splitted_is == True and sent_splitted_is == False:
        print('class is', 'text_splitted')
        return 'text_splitted'

    #sentenceの分割がある（sentenceの分割の有無は問わない）
    if sent_splitted_is == True:
        print('class is', 'sent_splitted')
        return 'sent_splitted'

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
#csvの読み込み
#-------------------------------------------
input_file_regex = INPUT_FILE_PREFIX + '*' + INPUT_FILE_SUFFIX
input_path_regex = str(output_dir.joinpath(input_file_regex))
input_path = select_file(input_path_regex)

if input_path == '':
    print('error :[', input_path_regex, '] is no match.')
    sys.exit()

df = pd.read_csv(input_path, dtype=str)
print('input file :', input_path)
print('df\n',df)

#-------------------------------------------
#解析 & データ分割
#-------------------------------------------
id_list_none, id_list_raw, id_list_txt_splt, id_list_snt_splt = [],[],[],[]

df_grp = df.groupby('PatternID',sort=False).count().copy()
id_list = df_grp.index.tolist()

nof_records = len(df)
nof_patterns = len(df.groupby('PatternID'))
nof_no_prereq_pattern = len(df_grp[df_grp['PrereqID'] == 0])
nof_with_prereq = nof_patterns - nof_no_prereq_pattern

for id in id_list:
    _class = pattern_classifier(id, df)
    if _class == 'no_prereq': id_list_none.append(id)
    if _class == 'raw': id_list_raw.append(id)
    if _class == 'text_splitted': id_list_txt_splt.append(id)
    if _class == 'sent_splitted': id_list_snt_splt.append(id)

nof_patterns_raw = len(id_list_raw)
nof_patterns_none = len(id_list_none)
nof_patterns_txt_splt = len(id_list_txt_splt)
nof_patterns_snt_splt = len(id_list_snt_splt)

per_patterns_raw = round((nof_patterns_raw/nof_with_prereq)*100)
per_patterns_txt_splt = round((nof_patterns_txt_splt/nof_with_prereq)*100)
per_patterns_snt_splt = round((nof_patterns_snt_splt/nof_with_prereq)*100)

#-------------------------------------------
#CSV出力
#-------------------------------------------
cols_output_none = ['PatternID','PrereqID','SentID','SplitID','Name','Prerequisite','SentLabel','Sentence','SplitLabel','Split']
df_output_none = df[df['PatternID'].isin(id_list_none)].loc[:,cols_output_none].reset_index(drop=True)
df_output_none.insert(0,'SubsetClass','no_prerequisite')

cols_output_raw = ['PatternID','PrereqID','SentID','SplitID','Name','Prerequisite','SentLabel','Sentence','SplitLabel','Split']
df_output_raw = df[df['PatternID'].isin(id_list_raw)].loc[:,cols_output_raw].reset_index(drop=True)
df_output_raw.insert(0,'SubsetClass','no_splitted')

cols_output_txt_splt = ['PatternID','PrereqID','SentID','SplitID','Name','Prerequisite','SentLabel','Sentence','SplitLabel','Split']
df_output_txt_splt = df[df['PatternID'].isin(id_list_txt_splt)].loc[:,cols_output_txt_splt].reset_index(drop=True)
df_output_txt_splt.insert(0,'SubsetClass','text_splitted')

cols_output_snt_splt = ['PatternID','PrereqID','SentID','SplitID','Name','Prerequisite','SentLabel','Sentence','SplitLabel','Split']
df_output_snt_splt = df[df['PatternID'].isin(id_list_snt_splt)].loc[:,cols_output_snt_splt].reset_index(drop=True)
df_output_snt_splt.insert(0,'SubsetClass','sentence_splitted')

df_output_all = pd.concat([df_output_none,df_output_raw,df_output_txt_splt,df_output_snt_splt], ignore_index=True)
df_output_all.insert(0,'ID',df_output_all.index.values.tolist())

print('df_output_none\n',df_output_none)
print('df_output_raw\n',df_output_raw)
print('df_output_txt_splt\n',df_output_txt_splt)
print('df_output_snt_splt\n',df_output_snt_splt)
print('df_output_all\n',df_output_all)

now = datetime.datetime.now()

output_file_name = OUTPUT_NOPRRQ_FILE_PREFIX + now.strftime('%Y%m%d_%H%M%S' + OUTPUT_FILE_SUFFIX)
output_path = os.path.join(output_dir, output_file_name)
df_output_none.to_csv(output_path)

output_file_name = OUTPUT_NOSPLT_FILE_PREFIX + now.strftime('%Y%m%d_%H%M%S' + OUTPUT_FILE_SUFFIX)
output_path = os.path.join(output_dir, output_file_name)
df_output_raw.to_csv(output_path)

output_file_name = OUTPUT_TXSPLT_FILE_PREFIX + now.strftime('%Y%m%d_%H%M%S' + OUTPUT_FILE_SUFFIX)
output_path = os.path.join(output_dir, output_file_name)
df_output_txt_splt.to_csv(output_path)

output_file_name = OUTPUT_STSPLT_FILE_PREFIX + now.strftime('%Y%m%d_%H%M%S' + OUTPUT_FILE_SUFFIX)
output_path = os.path.join(output_dir, output_file_name)
df_output_snt_splt.to_csv(output_path)

output_file_name = OUTPUT_ALL_FILE_PREFIX + now.strftime('%Y%m%d_%H%M%S' + OUTPUT_FILE_SUFFIX)
output_path = os.path.join(output_dir, output_file_name)
df_output_all.to_csv(output_path)

#-------------------------------------------
#結果出力
#-------------------------------------------
print()
print('records                          :', str(nof_records).zfill(4))
print('patterns                         :', str(nof_patterns).zfill(4))
print('patterns with prerequisites      :', str(nof_with_prereq).zfill(4))
print('patterns with raw text           :', str(nof_patterns_raw).zfill(4), '/', str(nof_with_prereq).zfill(4), str(per_patterns_raw).zfill(3)+'%')
print('patterns with splitted text      :', str(nof_patterns_txt_splt).zfill(4), '/', str(nof_with_prereq).zfill(4), str(per_patterns_txt_splt).zfill(3)+'%')
print('patterns with splitted sentences :', str(nof_patterns_snt_splt).zfill(4), '/', str(nof_with_prereq).zfill(4), str(per_patterns_snt_splt).zfill(3)+'%')

#-------------------------------------------
#end
#-------------------------------------------
print()
print(datetime.datetime.now())
print(__file__)
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('\n')
