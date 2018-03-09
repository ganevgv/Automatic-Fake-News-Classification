import numpy as np
import pandas as pd
import nltk
from collections import defaultdict
import os.path


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
              'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
              'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
              'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
              'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
              'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
              'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
              'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
              'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
              'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', '\n', 'the',
              '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=',
              '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
# stop words source: https://github.com/uclmr/stat-nlp-book/blob/python/chapters/doc_classify.ipynb


def tokenizer_nltk(input):
    return nltk.word_tokenize(input)


# sklearn models
def statement_to_dict(statement):
    statement_features = defaultdict(float)
    for token in statement:
        statement_features[token] += 1.0
    return statement_features


def build_statements_features(df, vectorizer, train=True, tokenizer=tokenizer_nltk):
    filtered_statements_dic = {}

    for index, row in df.iterrows():
        filtered_statement = []
        tokenized_statement = tokenizer(row['statement'].lower().decode('utf-8'))
        for token in tokenized_statement:
            if token not in stop_words:
                filtered_statement.append(token)

        filtered_statements_dic[index] = filtered_statement

    filtered_statements = filtered_statements_dic.values()

    if train:
        statements_features = vectorizer.fit_transform([statement_to_dict(statement) for statement in filtered_statements])
    else:
        statements_features = vectorizer.transform([statement_to_dict(statement) for statement in filtered_statements])

    return statements_features


# tensorflow models
def extract_vocab(df, embeddings, tokenizer=tokenizer_nltk):
    path = '../saved_data/embeddings/'

    lowercase = True if embeddings in ['glove_100d_6b', 'glove_300d_6b', 'facebook'] else False

    if os.path.isfile(path + 'vocab.txt') and lowercase:
        df_vocab = pd.read_table(path + 'vocab.txt', sep=' ', header=None, index_col=0)
        print('full vocab already exists - vocab loaded, tokens found: {:.0f}'.format(len(df_vocab) - 2))

    elif os.path.isfile(path + 'vocab_upper.txt') and not lowercase:
        df_vocab = pd.read_table(path + 'vocab_upper.txt', sep=' ', header=None, index_col=0)
        print('full upper vocab already exists - upper vocab loaded, tokens found: {:.0f}'.format(len(df_vocab) - 2))

    else:
        df_vocab = pd.DataFrame(columns=['1'])
        # add OOV and PAD tokens
        df_vocab.loc['PAD'] = 0
        df_vocab.loc['OOV'] = 1

        if lowercase:
            combined_statements = ' '.join(df['statement']).lower().decode('utf-8')
        else:
            combined_statements = ' '.join(df['statement']).decode('utf-8')

        tokenized_combined_statements = tokenizer(combined_statements)
        for token in tokenized_combined_statements:
            token = token.encode('utf-8')
            if token not in df_vocab.index:
                df_vocab.loc[token] = len(df_vocab)
        df_vocab = df_vocab.astype(int)
        df_vocab.to_csv(path + 'vocab.txt', sep=' ', header=None, index_col=0)
        print('full vocab built and saved, tokens found: {:.0f}'.format(len(df_vocab) - 2))

    return df_vocab


def load_embeddings(df, df_vocab, embeddings):
    # glove_300d_6b
    if embeddings == 'glove_300d_6b':
        path = '../saved_data/embeddings/glove_300d_6b/'

        # check if embeddings reduced exist already and if so return them
        if os.path.isfile(path + 'glove_300d_6b_reduced.txt') and os.path.isfile(path + 'vocab_reduced.txt'):
            df_embeddings_reduced = pd.read_table(path + 'glove_300d_6b_reduced.txt', sep=' ', header=None, index_col=0)
            df_vocab_reduced = pd.read_table(path + 'vocab_reduced.txt', sep=' ', header=None, index_col=0)
            print('glove_300d_6b reduced embeddings and reduced vocab already exist - glove_300d_6b reduced embeddings loaded')
        # build the vocab from the combined statements and reduce the embeddings
        else:
            df_embeddings = pd.read_table(path + 'glove_300d_6b.txt', sep=' ', header=None, index_col=0)
            df_embeddings_reduced = pd.DataFrame(columns=df_embeddings.columns)
            df_embeddings_reduced.loc['PAD'] = 0
            df_embeddings_reduced.loc['OOV'] = 1
            print('full glove_300d_6b embeddings loaded')

            for token in df_vocab.index:
                # token = token.encode('utf-8')
                if token in df_embeddings.index:
                    df_embeddings_reduced = df_embeddings_reduced.append(df_embeddings.loc[token], ignore_index=False)
                # else:
                #     df_vocab.drop(df_vocab.loc[token])

            # change OOV token
            df_embeddings_reduced.loc['OOV'] = df_embeddings_reduced.iloc[2:].mean(axis=0)

            df_embeddings_reduced.to_csv(path + 'glove_300d_6b_reduced.txt', sep=' ', header=None, index_col=0)
            print('glove_300d_6b reduced embeddings built and saved, tokens found: {:.0f}, out of: {:.0f}'.format(len(df_embeddings_reduced) - 2, len(df_vocab) - 2))

            df_vocab_reduced = pd.DataFrame(columns=['1'])
            for i, token in enumerate(df_embeddings_reduced.index):
                df_vocab_reduced.loc[token] = i
            df_vocab_reduced = df_vocab_reduced.astype(int)
            df_vocab_reduced.to_csv(path + 'vocab_reduced.txt', sep=' ', header=None, index_col=0)
            print('vocab reduced built and saved')

    # glove_100d_6b
    if embeddings == 'glove_100d_6b':
        path = '../saved_data/embeddings/glove_100d_6b/'

        # check if embeddings reduced exist already and if so return them
        if os.path.isfile(path + 'glove_100d_6b_reduced.txt') and os.path.isfile(path + 'vocab_reduced.txt'):
            df_embeddings_reduced = pd.read_table(path + 'glove_100d_6b_reduced.txt', sep=' ', header=None, index_col=0)
            df_vocab_reduced = pd.read_table(path + 'vocab_reduced.txt', sep=' ', header=None, index_col=0)
            print('glove_100d_6b reduced embeddings and reduced vocab already exist - glove_100d_6b reduced embeddings loaded')
        # build the vocab from the combined statements and reduce the embeddings
        else:
            df_embeddings = pd.read_table(path + 'glove_100d_6b.txt', sep=' ', header=None, index_col=0)
            df_embeddings_reduced = pd.DataFrame(columns=df_embeddings.columns)
            df_embeddings_reduced.loc['PAD'] = 0
            df_embeddings_reduced.loc['OOV'] = 1
            print('full glove_100d_6b embeddings loaded')

            for token in df_vocab.index:
                # token = token.encode('utf-8')
                if token in df_embeddings.index:
                    df_embeddings_reduced = df_embeddings_reduced.append(df_embeddings.loc[token], ignore_index=False)
                # else:
                #     df_vocab.drop(df_vocab.loc[token])

            # change OOV token
            df_embeddings_reduced.loc['OOV'] = df_embeddings_reduced.iloc[2:].mean(axis=0)

            df_embeddings_reduced.to_csv(path + 'glove_100d_6b_reduced.txt', sep=' ', header=None, index_col=0)
            print('glove_100d_6b reduced embeddings built and saved, tokens found: {:.0f}, out of: {:.0f}'.format(len(df_embeddings_reduced) - 2, len(df_vocab) - 2))

            df_vocab_reduced = pd.DataFrame(columns=['1'])
            for i, token in enumerate(df_embeddings_reduced.index):
                df_vocab_reduced.loc[token] = i
            df_vocab_reduced = df_vocab_reduced.astype(int)
            df_vocab_reduced.to_csv(path + 'vocab_reduced.txt', sep=' ', header=None, index_col=0)
            print('vocab reduced built and saved')

    # glove_300d_6b
    if embeddings == 'glove_300d_84b':
        path = '../saved_data/embeddings/glove_300d_84b/'

        # check if embeddings reduced exist already and if so return them
        if os.path.isfile(path + 'glove_300d_84b_reduced.txt') and os.path.isfile(path + 'vocab_upper_reduced.txt'):
            df_embeddings_reduced = pd.read_table(path + 'glove_300d_84b_reduced.txt', sep=' ', header=None, index_col=0)
            df_vocab_reduced = pd.read_table(path + 'vocab_upper_reduced.txt', sep=' ', header=None, index_col=0)
            print('glove_300d_84b reduced embeddings and reduced upper vocab already exist - glove_300d_84b reduced embeddings loaded')
        # build the vocab from the combined statements and reduce the embeddings
        else:
            df_embeddings = pd.read_table(path + 'glove_300d_84b.txt', sep=' ', header=None, index_col=0)
            df_embeddings_reduced = pd.DataFrame(columns=df_embeddings.columns)
            df_embeddings_reduced.loc['PAD'] = 0
            df_embeddings_reduced.loc['OOV'] = 1
            print('full glove_300d_84b embeddings loaded')

            for token in df_vocab.index:
                # token = token.encode('utf-8')
                if token in df_embeddings.index:
                    df_embeddings_reduced = df_embeddings_reduced.append(df_embeddings.loc[token], ignore_index=False)
                # else:
                #     df_vocab.drop(df_vocab.loc[token])

            # change OOV token
            df_embeddings_reduced.loc['OOV'] = df_embeddings_reduced.iloc[2:].mean(axis=0)

            df_embeddings_reduced.to_csv(path + 'glove_300d_84b_reduced.txt', sep=' ', header=None, index_col=0)
            print('glove_300d_84b reduced embeddings built and saved, tokens found: {:.0f}, out of: {:.0f}'.format(len(df_embeddings_reduced) - 2, len(df_vocab) - 2))

            df_vocab_reduced = pd.DataFrame(columns=['1'])
            for i, token in enumerate(df_embeddings_reduced.index):
                df_vocab_reduced.loc[token] = i
            df_vocab_reduced = df_vocab_reduced.astype(int)
            df_vocab_reduced.to_csv(path + 'vocab_upper_reduced.txt', sep=' ', header=None, index_col=0)
            print('vocab upper reduced built and saved')

    # google
    if embeddings == 'google':
        path = '../saved_data/embeddings/google/'

        # check if embeddings reduced exist already and if so return them
        if os.path.isfile(path + 'google_word2vec_300d_reduced.txt') and os.path.isfile(path + 'vocab_upper_reduced.txt'):
            df_embeddings_reduced = pd.read_table(path + 'google_word2vec_300d_reduced.txt', sep=' ', header=None, index_col=0)
            df_vocab_reduced = pd.read_table(path + 'vocab_upper_reduced.txt', sep=' ', header=None, index_col=0)
            print('google_word2vec_300d reduced embeddings and reduced upper vocab already exist - google_word2vec_300d reduced embeddings loaded')
        # build the vocab from the combined statements and reduce the embeddings
        else:
            df_embeddings = pd.read_table(path + 'google_word2vec_300d.txt', sep=' ', header=None, index_col=0, nrows=1000000)
            df_embeddings_reduced = pd.DataFrame(columns=df_embeddings.columns)
            df_embeddings_reduced.loc['PAD'] = 0
            df_embeddings_reduced.loc['OOV'] = 1
            print('full google_word2vec_300d embeddings loaded')

            for token in df_vocab.index:
                # token = token.encode('utf-8')
                if token in df_embeddings.index:
                    df_embeddings_reduced = df_embeddings_reduced.append(df_embeddings.loc[token], ignore_index=False)
                # else:
                #     df_vocab.drop(df_vocab.loc[token])

            # change OOV token
            df_embeddings_reduced.loc['OOV'] = df_embeddings_reduced.iloc[2:].mean(axis=0)

            df_embeddings_reduced.to_csv(path + 'google_word2vec_300d_reduced.txt', sep=' ', header=None, index_col=0)
            print('google_word2vec_300d reduced embeddings built and saved, tokens found: {:.0f}, out of: {:.0f}'.format(len(df_embeddings_reduced) - 2, len(df_vocab) - 2))

            df_vocab_reduced = pd.DataFrame(columns=['1'])
            for i, token in enumerate(df_embeddings_reduced.index):
                df_vocab_reduced.loc[token] = i
            df_vocab_reduced = df_vocab_reduced.astype(int)
            df_vocab_reduced.to_csv(path + 'vocab_upper_reduced.txt', sep=' ', header=None, index_col=0)
            print('vocab upper reduced built and saved')

    # facebook
    if embeddings == 'facebook':
        path = '../saved_data/embeddings/facebook/'

        # check if embeddings reduced exist already and if so return them
        if os.path.isfile(path + 'facebook_fastText_300d_reduced.txt') and os.path.isfile(path + 'vocab_reduced.txt'):
            df_embeddings_reduced = pd.read_table(path + 'facebook_fastText_300d_reduced.txt', sep=' ', header=None, index_col=0)
            df_vocab_reduced = pd.read_table(path + 'vocab_reduced.txt', sep=' ', header=None, index_col=0)
            print('facebook_fastText_300d reduced embeddings and reduced upper vocab already exist - facebook_fastText_300d reduced embeddings loaded')
        # build the vocab from the combined statements and reduce the embeddings
        else:
            df_embeddings = pd.read_table(path + 'wiki.en.vec', sep=' ', skiprows=1, header=None, index_col=0, usecols=range(301))
            df_embeddings_reduced = pd.DataFrame(columns=df_embeddings.columns)
            df_embeddings_reduced.loc['PAD'] = 0
            df_embeddings_reduced.loc['OOV'] = 1
            print('full facebook_fastText_300d embeddings loaded')

            for token in df_vocab.index:
                # token = token.encode('utf-8')
                if token in df_embeddings.index:
                    df_embeddings_reduced = df_embeddings_reduced.append(df_embeddings.loc[token], ignore_index=False)
                # else:
                #     df_vocab.drop(df_vocab.loc[token])

            # change OOV token
            df_embeddings_reduced.loc['OOV'] = df_embeddings_reduced.iloc[2:].mean(axis=0)

            df_embeddings_reduced.to_csv(path + 'facebook_fastText_300d_reduced.txt', sep=' ', header=None, index_col=0)
            print('facebook_fastText_300d reduced embeddings built and saved, tokens found: {:.0f}, out of: {:.0f}'.format(len(df_embeddings_reduced) - 2, len(df_vocab) - 2))

            df_vocab_reduced = pd.DataFrame(columns=['1'])
            for i, token in enumerate(df_embeddings_reduced.index):
                df_vocab_reduced.loc[token] = i
            df_vocab_reduced = df_vocab_reduced.astype(int)
            df_vocab_reduced.to_csv(path + 'vocab_reduced.txt', sep=' ', header=None, index_col=0)
            print('vocab reduced built and saved')

    return df_embeddings_reduced, df_vocab_reduced


def build_W_embeddings(df_embeddings_reduced):
    W = df_embeddings_reduced.as_matrix()
    return W


def build_statements_embeddings(df, df_vocab, tokenizer=tokenizer_nltk, max_statement_len=None):
    embedded_statements_dic = {}
    max_statement_len_ = -1

    for index, row in df.iterrows():
        embedded_statement = []
        tokenized_statement = tokenizer(row['statement'].lower().decode('utf-8'))
        for token in tokenized_statement:
            if token in df_vocab.index:
                embedded_statement.append(np.array(df_vocab.loc[token]))
            else:
                embedded_statement.append(np.array(df_vocab.loc['OOV']))

        embedded_statements_dic[index] = np.array(embedded_statement)
        if len(embedded_statement) > max_statement_len_:
            max_statement_len_ = len(embedded_statement)

    if max_statement_len is not None:
        max_statement_len_ = max_statement_len

    for key in embedded_statements_dic:
        embedded_statement = embedded_statements_dic[key]
        embedded_statement_len = np.shape(embedded_statement)[0]
        padded_embedded_statement = np.tile(np.array(df_vocab.loc['PAD']), (max_statement_len_, 1))

        if max_statement_len_ >= embedded_statement_len:
            padded_embedded_statement[:embedded_statement_len, :] = embedded_statement
        else:
            padded_embedded_statement[:, :] = embedded_statement[:max_statement_len_, :]
        embedded_statements_dic[key] = padded_embedded_statement

    embedded_statements_matrix = np.squeeze(np.asarray(embedded_statements_dic.values()))

    return embedded_statements_matrix, max_statement_len_
