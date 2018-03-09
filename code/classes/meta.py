import numpy as np
import pandas as pd
from embeddings import tokenizer_nltk
import os.path


def extract_history(df, remove_history=0, one_hot=True, new_speakers_label=None):
    label_list = df['label'].unique()

    df_history = df.drop_duplicates(subset=['speaker'], keep='first')
    df_history = df_history[['speaker']]

    for label in label_list:
        label_data = df[df['label'] == label]
        labels_counts = label_data.groupby('speaker').agg({'label': len})
        labels_counts_dic = labels_counts['label']
        df_history[label] = df_history['speaker'].map(labels_counts_dic)
        df_history[label].fillna(0, inplace=True)

    df_history = df_history[df_history.sum(axis=1) >= remove_history]

    if one_hot:
        max_label = df_history[label_list].idxmax(axis=1)
        df_history[label_list] = 0
        for index, _ in df_history.iterrows():
            df_history.loc[index, max_label[index]] = 1
    else:
        row_sum = df_history.sum(axis=1)
        for label in label_list:
            df_history[label] = df_history[label] / row_sum

    df_history = df_history.set_index('speaker', drop=True)

    df_history.loc['OOV', label_list] = 0
    if new_speakers_label is not None:
        df_history.loc['OOV', new_speakers_label] = 1

    return df_history


def build_speakers_credit(df, df_history):
    speakers_credit_dic = {}

    for index, row in df.iterrows():
        if row['speaker'] in df_history.index:
            speakers_credit_dic[index] = df_history.loc[row['speaker']].values
        else:
            speakers_credit_dic[index] = df_history.loc['OOV'].values

    speakers_credit_matrix = np.asarray(speakers_credit_dic.values())

    return speakers_credit_matrix


def extract_meta_vocab(df, tokenizer=tokenizer_nltk, meta_columns=['subjects', 'speaker', "speaker's job", 'state', 'party', 'context']):
    path = '../saved_data/meta/'

    # tokenize (currently lower(), tokenizer_nltk)
    if os.path.isfile(path + 'vocab.txt'):
        df_meta_vocab = pd.read_table(path + 'vocab.txt', sep=' ', header=None, index_col=0)
        print('full meta vocab already exists - meta vocab loaded, tokens found: {:.0f}'.format(len(df_meta_vocab) - 2))

    else:
        df_meta_vocab = pd.DataFrame(columns=['1'])
        # add OOV and PAD tokens
        df_meta_vocab.loc['PAD'] = 0
        df_meta_vocab.loc['OOV'] = 1

        combined_meta_data = []
        for meta_column in meta_columns:
            df[meta_column] = df[meta_column].fillna('NaN_' + meta_column)
            meta_data = ' '.join(df[meta_column]).lower().decode('utf-8')
            combined_meta_data.append(meta_data)
        combined_meta_data = ' '.join(meta_data for meta_data in combined_meta_data)

        tokenized_combined_meta_data = tokenizer(combined_meta_data)
        for token in tokenized_combined_meta_data:
            token = token.encode('utf-8')
            if token not in df_meta_vocab.index:
                df_meta_vocab.loc[token] = len(df_meta_vocab)
        df_meta_vocab = df_meta_vocab.astype(int)
        df_meta_vocab.to_csv(path + 'vocab.txt', sep=' ', header=None, index_col=0)
        print('full meta vocab built and saved, tokens found: {:.0f}'.format(len(df_meta_vocab) - 2))

    return df_meta_vocab


def build_W_meta(df_meta_vocab):
    W = df_meta_vocab.as_matrix()
    return W


def build_meta_embeddings(df, df_meta_vocab, tokenizer=tokenizer_nltk, meta_columns=['subjects', 'speaker', "speaker's job", 'state', 'party', 'context'], max_meta_len=None):
    # meta_columns = ['speaker']
    embedded_metas_dic = {}
    max_meta_len_ = -1

    for meta_column in meta_columns:
        df[meta_column] = df[meta_column].fillna('NaN_' + meta_column)

    for index, row in df.iterrows():
        embedded_metas = []
        for meta_column in meta_columns:
            embedded_meta = []
            tokenized_meta_column = tokenizer(row[meta_column].lower().decode('utf-8'))
            for token in tokenized_meta_column:
                if token == ',':
                    pass
                else:
                    token = token.encode('utf-8')
                    if token in df_meta_vocab.index:
                        embedded_meta.append(np.array(df_meta_vocab.loc[token]))
                    else:
                        embedded_meta.append(np.array(df_meta_vocab.loc['OOV']))
            embedded_metas.append(embedded_meta)

            if len(embedded_meta) > max_meta_len_:
                max_meta_len_ = len(embedded_meta)

        embedded_metas_dic[index] = np.array(embedded_metas)

    if max_meta_len is not None:
        max_meta_len_ = max_meta_len

    speakers_credit_matrix = np.full([len(embedded_metas_dic), len(meta_columns), max_meta_len_], np.array(df_meta_vocab.loc['PAD']), dtype=np.int32)

    for i, embedded_metas in embedded_metas_dic.iteritems():
        for j, embedded_meta in enumerate(embedded_metas):
            speakers_credit_matrix[i, j, :len(embedded_meta)] = embedded_meta
    speakers_credit_matrix = np.asarray(speakers_credit_matrix)

    return speakers_credit_matrix, max_meta_len_
