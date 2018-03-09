import numpy as np
import nltk
from embeddings import tokenizer_nltk


pos_tags = ['PAD', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP$', 'PRP', 'WP', 'WP$', 'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS',
            'RB', 'RBR', 'RBS', 'WRB', 'DT', 'PDT', 'WDT', 'SYM', 'POS', 'LRB', 'RRB', ',', '-', ':', ';', '.', '``', '"', '$', 'CD', 'DAT',
            'CC', 'EX', 'FW', 'IN', 'RP', 'TO', 'UH', 'URL', 'USER', 'EMAIL', 'NNPOS', 'UR', ':)', 'UH', '#', '!!!', '...', '\'\'', '(', ')', 'LS'
            ]


def pos_index(labels, classes=pos_tags):
    indexed_labels = list(map(lambda x: int(classes.index(x)), labels))
    return indexed_labels


def build_W_pos(classes=pos_tags):
    n_classes = len(classes)
    W = np.zeros((n_classes, n_classes - 1))
    W[1:, :] = np.identity(n_classes - 1)
    return W


def build_statements_pos(df, max_statement_len, tokenizer=tokenizer_nltk):
    pos_statements_dic = {}

    for index, row in df.iterrows():
        pos_statement = []
        tokenized_statement = tokenizer(row['statement'].decode('utf-8'))
        pos_statement = [token[1] for token in nltk.pos_tag(tokenized_statement)]
        pos_statement = pos_index(pos_statement)

        pos_statement_len = np.shape(pos_statement)[0]
        padded_pos_statement = np.tile(np.array(pos_tags.index('PAD')), (max_statement_len))

        if max_statement_len >= pos_statement_len:
            padded_pos_statement[:pos_statement_len] = pos_statement
        else:
            padded_pos_statement[:] = pos_statement[:max_statement_len]
        pos_statements_dic[index] = padded_pos_statement

    pos_statements_matrix = np.asarray(pos_statements_dic.values())

    return pos_statements_matrix
