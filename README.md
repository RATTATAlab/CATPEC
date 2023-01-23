# 概要
CAPECの攻撃パターンをアタックツリーのパーツに変換するスクリプト．[RATTATA](https://github.com/RATTATAlab/RATTATA)にインポートして使う．



# セットアップ

- CAPECデータファイルの設置
    - 最新のCAPECデータファイルをダウンロードする．URL:[https://capec.mitre.org/data/xml/views/1000.xml.zip](https://capec.mitre.org/data/xml/views/1000.xml.zip)
    - data/capec/に保存する．
    - src/1_dump_and_normalize.pyのソースコードに合わせてファイルをリネームする．
- Stanford Parserの設置
    - アプリケーションと英語の言語モデルをダウンロードする．URL:[https://nlp.stanford.edu/software/lex-parser.shtml](https://nlp.stanford.edu/software/lex-parser.shtml)
    - tools/stanford-parserに保存する
        - フォルダ内が以下のようになればOK．（バージョン4.2.0の場合）
            - stanford-corenlp-4.2.0-models-english.jar
            - stanford-parser-full-2020-11-17/



# 使用方法
src/内を1から9まで順に実行しておけばOK．途中の出力ファイルはdata/intermediate/へ，成果物のアタックツリーパーツリストはdata/processed/へ出力される．