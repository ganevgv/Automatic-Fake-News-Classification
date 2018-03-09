import numpy as np
from sklearn.feature_extraction import DictVectorizer
from data_munging import preprocess_data, load_data
from embeddings import build_statements_features, extract_vocab, load_embeddings, build_W_embeddings, build_statements_embeddings
from pos import build_W_pos, build_statements_pos
from meta import extract_history, build_speakers_credit, extract_meta_vocab, build_W_meta, build_meta_embeddings


def preprocess_import_data(binary_data):
    preprocess_data()
    if binary_data:
        train_data = load_data('train_binirized')
        valid_data = load_data('valid_binirized')
        test_data = load_data('test_binirized')
    else:
        train_data = load_data('train')
        valid_data = load_data('valid')
        test_data = load_data('test')

    return train_data, valid_data, test_data


def import_majority_data(binary_data=False):
    train_data, valid_data, test_data = preprocess_import_data(binary_data)

    X_train = train_data
    X_valid = valid_data
    X_test = test_data

    y_train = train_data['label']
    y_valid = valid_data['label']
    y_test = test_data['label']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def import_sklearn_data(binary_data=False):
    train_data, valid_data, test_data = preprocess_import_data(binary_data)

    vectorizer = DictVectorizer()
    X_train = build_statements_features(train_data, vectorizer)
    X_valid = build_statements_features(valid_data, vectorizer, train=False)
    X_test = build_statements_features(test_data, vectorizer, train=False)

    y_train = train_data['label']
    y_valid = valid_data['label']
    y_test = test_data['label']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def one_hot(labels, classes=load_data('train')['label'].unique().tolist()):
    n_classes = len(classes)
    n_samples = len(labels)
    indexed_labels = list(map(lambda x: int(classes.index(x)), labels))
    one_hot_labels = np.zeros([n_samples, n_classes]).astype(int)
    one_hot_labels[tuple(range(n_samples)), tuple(indexed_labels)] = 1
    return one_hot_labels


def unone_hotify(labels, classes=load_data('train')['label'].unique().tolist()):
    labels_flat = np.argmax(labels, axis=1)
    return [classes[i] for i in labels_flat]


def import_tensorflow_data(statement=True, pos=False, meta_all=False, binary_data=False, embeddings='glove_300d_6b'):
    train_data, valid_data, test_data = preprocess_import_data(binary_data)
    X_train = {}
    X_valid = {}
    X_test = {}

    if statement:
        df_vocab = extract_vocab(train_data, embeddings)
        df_embeddings, df_vocab_reduced = load_embeddings(train_data, df_vocab, embeddings)
        X_train['W_embeddings'] = build_W_embeddings(df_embeddings)
        X_train['embedded_statement'], max_statement_len = build_statements_embeddings(train_data, df_vocab_reduced)
        X_valid['embedded_statement'], _ = build_statements_embeddings(valid_data, df_vocab_reduced, max_statement_len=max_statement_len)
        X_test['embedded_statement'], _ = build_statements_embeddings(test_data, df_vocab_reduced, max_statement_len=max_statement_len)

    if pos:
        X_train['W_pos'] = build_W_pos()
        X_train['pos_statement'] = build_statements_pos(train_data, max_statement_len=max_statement_len)
        X_valid['pos_statement'] = build_statements_pos(valid_data, max_statement_len=max_statement_len)
        X_test['pos_statement'] = build_statements_pos(test_data, max_statement_len=max_statement_len)

    if meta_all:
        # df_history = extract_history(train_data, remove_history=10, one_hot=False)
        # X_train['meta_history'] = build_speakers_credit(train_data, df_history)
        # X_valid['meta_history'] = build_speakers_credit(valid_data, df_history)
        # X_test['meta_history'] = build_speakers_credit(test_data, df_history)

        df_meta_vocab = extract_meta_vocab(train_data)
        X_train['W_meta'] = build_W_meta(df_meta_vocab)
        X_train['meta_all'], max_meta_len = build_meta_embeddings(train_data, df_meta_vocab)
        X_valid['meta_all'], _ = build_meta_embeddings(valid_data, df_meta_vocab, max_meta_len=max_meta_len)
        X_test['meta_all'], _ = build_meta_embeddings(test_data, df_meta_vocab, max_meta_len=max_meta_len)

    # elif meta_history:
    #     df_history = extract_history(train_data, remove_history=1, one_hot=False, new_speakers_label='false')
    #     X_train['meta_history'] = build_speakers_credit(train_data, df_history)
    #     X_valid['meta_history'] = build_speakers_credit(valid_data, df_history)
    #     X_test['meta_history'] = build_speakers_credit(test_data, df_history)

    # elif meta_subject:
    # elif meta_speaker:
    # elif meta_job:
    # elif meta_state:
    # elif meta_party:
    # elif meta_context:

    classes = train_data['label'].unique().tolist()
    y_train = one_hot(train_data['label'].tolist(), classes)
    y_valid = one_hot(valid_data['label'].tolist(), classes)
    # test_data = test_data.replace({True: 'true', False: 'false'})
    y_test = one_hot(test_data['label'].tolist(), classes)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
