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
INPUT_FILE_PREFIX = 'summaraized_all_subset_'
INPUT_FILE_SUFFIX = '.csv'
OUTPUT_FILE_PREFIX = 'parts_list'
OUTPUT_FILE_SUFFIX = '.xml'
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
output_dir = data_dir.joinpath('processed')

#-------------------------------------------
#output用のXMLを用意
#-------------------------------------------
output_xml_root = ET.Element('Outputs')
output_xml_patterns = ET.Element('Patterns')
output_xml_root.append(output_xml_patterns)

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
#convert csv to xml
#-------------------------------------------
df_grp = df.groupby('PatternID',sort=False).count().copy()
id_list = df_grp.index.values.tolist()

for id in id_list:
    name_list = df[df['PatternID'] == id]['Name'].values
    ad_list_raw = df[(df['PatternID'] == id) & (df['Class'] == 'adversary')]['Prerequisite'].values
    ta_list_raw = df[(df['PatternID'] == id) & (df['Class'] == 'target')]['Prerequisite'].values
    ad_list_ja = df[(df['PatternID'] == id) & (df['Class'] == 'adversary')]['Prerequisite_JA'].values
    ta_list_ja = df[(df['PatternID'] == id) & (df['Class'] == 'target')]['Prerequisite_JA'].values

    output_xml_pattern = ET.Element('Pattern')
    output_xml_pattern.set('ID', id)
    output_xml_pattern.set('Name', name_list[0])
    output_xml_patterns.append(output_xml_pattern)

    if len(ad_list_raw) == 0 and len(ta_list_raw) == 0:continue

    output_xml_prereqs = ET.Element('Prerequisites')
    output_xml_pattern.append(output_xml_prereqs)

    for i, prereq in enumerate(ad_list_raw):
        output_xml_prereq = ET.Element('Prerequisite')
        output_xml_prereq.set('Class', 'adversary')
        output_xml_prereqs.append(output_xml_prereq)

        output_xml_raw = ET.Element('Raw')
        output_xml_raw.text = prereq
        output_xml_prereq.append(output_xml_raw)

        output_xml_ja = ET.Element('Japanese')
        output_xml_ja.text = ad_list_ja[i]
        output_xml_prereq.append(output_xml_ja)

    for j, prereq in enumerate(ta_list_raw):
        output_xml_prereq = ET.Element('Prerequisite')
        output_xml_prereq.set('Class', 'target')
        output_xml_prereqs.append(output_xml_prereq)

        output_xml_raw = ET.Element('Raw')
        output_xml_raw.text = prereq
        output_xml_prereq.append(output_xml_raw)

        output_xml_ja = ET.Element('Japanese')
        output_xml_ja.text = ta_list_ja[j]
        output_xml_prereq.append(output_xml_ja)

#-------------------------------------------
#結果出力
#-------------------------------------------
output_tree = ET.ElementTree(output_xml_root)
now = datetime.datetime.now()
output_file_name = OUTPUT_FILE_PREFIX + now.strftime('%Y%m%d_%H%M%S' + OUTPUT_FILE_SUFFIX)
output_path = str(output_dir.joinpath(output_file_name))

with open(output_path, 'wb') as output:
    output_tree.write(output, encoding=OUTPUT_ENCODING, xml_declaration=True)

#-------------------------------------------
#end
#-------------------------------------------
print()
print(datetime.datetime.now())
print(__file__)
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('\n')
