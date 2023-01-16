from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import math

class NaiveBayesClassifier:

    df_train_set = pd.DataFrame()
    df_vec_set = pd.DataFrame()
    df_summary = pd.DataFrame()

    def __init__(self):
        cols_df_train_set = ['ClassName','Texts']
        self.df_train_set = pd.DataFrame(index=[], columns=cols_df_train_set)

        cols_df_vec_set = ['ClassName','Vectors']
        self.df_vec_set = pd.DataFrame(index=[], columns=cols_df_vec_set)

        cols_df_summary = ['ClassName','NofTexts', 'NofWords', 'Vocabulary']
        self.df_summary = pd.DataFrame(index=[], columns=cols_df_summary)

    def append_train_set(self, class_name, text_list):
        _new_record = pd.Series([class_name, text_list], index=self.df_train_set.columns)
        self.df_train_set = pd.concat([self.df_train_set, pd.DataFrame([_new_record])], ignore_index=True)

    def transform_train(self, class_name, texts):
        vec_count = CountVectorizer()
        vec_count.fit(texts)
        x = vec_count.transform(texts)
        df = pd.DataFrame(x.toarray(), columns=vec_count.get_feature_names_out())

        _new_record = pd.Series([class_name, df], index=self.df_vec_set.columns)
        self.df_vec_set = pd.concat([self.df_vec_set, pd.DataFrame([_new_record])], ignore_index=True)

        nof_texts = len(df)
        nof_words = df.sum().sum()
        vocaburary = len(vec_count.vocabulary_)

        _new_record = pd.Series([class_name, nof_texts, nof_words, vocaburary], index=self.df_summary.columns)
        self.df_summary = pd.concat([self.df_summary, pd.DataFrame([_new_record])], ignore_index=True)

    def train(self):
        combine_class = ''
        combine_texts = []

        for _index, _record in self.df_train_set.iterrows():
            self.transform_train(_record['ClassName'], _record['Texts'])
            combine_class = combine_class + '_' + _record['ClassName']
            combine_texts = combine_texts + _record['Texts']

        self.transform_train(combine_class, combine_texts)

    def clac_p(self, df, class_name, logp_cl, numerator):
        logp_word_cl_sum = 0

        for _index_vec, _record_vec in self.df_vec_set.iterrows():
            if _record_vec['ClassName'] == class_name:

                for _column, _count in df.iloc[0].items():
                    if _count >= 1:
                        if _column in _record_vec['Vectors'].columns:
                            logp_word_cl = math.log((_record_vec['Vectors'][_column].sum() + 1) / numerator)
                        else:
                            logp_word_cl = math.log(1 / numerator)

                        logp_word_cl_sum = logp_word_cl_sum + logp_word_cl

        return logp_cl + logp_word_cl_sum

    def decision_class(self, prob_cl_list):
        max_p = 0
        class_name = ''
        for prob_cl in prob_cl_list:
#            print(prob_cl[0], prob_cl[1])
            if max_p == 0 or max_p < prob_cl[1]:
                max_p = prob_cl[1]
                class_name = prob_cl[0]

        return class_name

    def classify(self, text):
        vec_count = CountVectorizer()
        vec_count.fit([text])
        x = vec_count.transform([text])
        df = pd.DataFrame(x.toarray(), columns=vec_count.get_feature_names_out())

        prob_cl_list = []
        for _index_summary, _record_summary in self.df_summary.iterrows():
            if _index_summary == len(self.df_summary) -1:break

            class_name = _record_summary['ClassName']
            logp_cl = math.log(_record_summary['NofTexts']/self.df_summary.iloc[-1]['NofTexts'])
            numerator = _record_summary['NofWords'] + self.df_summary.iloc[-1]['Vocabulary']

            prob_cl_list.append((class_name,self.clac_p(df, class_name, logp_cl, numerator)))

        return self.decision_class(prob_cl_list)

    def get_train_set(self):
        return self.df_train_set

    def get_train_shape(self):
        return self.df_train_set.shape

    def get_train_columns(self):
        return self.df_train_set.columns

    def get_vec_set(self):
        return self.df_vec_set

    def get_vec(self, class_name):
        for _index, _record in self.df_vec_set.iterrows():
            if _record['ClassName'] == class_name:
                return _record['Vectors']
        return ''

    def get_summary(self):
        return self.df_summary
