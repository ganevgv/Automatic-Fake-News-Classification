import pandas as pd

data_sets = ['train', 'valid', 'test']
header_names = ['ID', 'label', 'statement', 'subjects', 'speaker', "speaker's job", 'state', 'party', 'barely-true-counts',
                'false-counts', 'half-true-counts', 'mostly-true-counts', 'pants-fire-counts', 'context'
                ]
# labels - ['false' 'half-true' 'mostly-true' 'true' 'barely-true' 'pants-fire']


def preprocess_data():
    data_sets_sizes = [10269, 1284, 1283]

    for i, data_set in enumerate(data_sets):
        file_path = '../data/' + data_set + '.tsv'
        with open(file_path, 'r') as file:
            text = file.read()
            text = text.replace("''", "'")
            text = text.replace('"', "'")

        with open(file_path, 'w') as file:
            file.write(text)

        data = pd.read_table(file_path, sep='\t', names=header_names)
        assert (len(data) == data_sets_sizes[i])


def binirize_labels():
    for data_set in data_sets:
        file_path = '../data/' + data_set + '.tsv'
        df = pd.read_table(file_path, sep='\t', names=header_names, index_col=0)
        df.loc[df['label'] == 'half-true', 'label'] = 'true'
        df.loc[df['label'] == 'mostly-true', 'label'] = 'true'

        df.loc[df['label'] == 'barely-true', 'label'] = 'false'
        df.loc[df['label'] == 'pants-fire', 'label'] = 'false'

        save_file_path = 'data/' + data_set + '_binirized.tsv'
        df.to_csv(save_file_path, sep='\t', header=None, index_col=0)


def load_data(data):
    file_path = '../data/'

    if data == 'train':
        train_data = pd.read_table(file_path + 'train.tsv', sep='\t', names=header_names)
        total_data = [train_data]

    if data == 'train_binirized':
        train_data = pd.read_table(file_path + 'train_binirized.tsv', sep='\t', names=header_names)
        total_data = [train_data]

    if data == 'valid':
        dev_data = pd.read_table(file_path + 'valid.tsv', sep='\t', names=header_names)
        total_data = [dev_data]

    if data == 'valid_binirized':
        dev_data = pd.read_table(file_path + 'valid_binirized.tsv', sep='\t', names=header_names)
        total_data = [dev_data]

    # if data == 'test':
    #     test_data = pd.read_table(file_path + 'test.tsv', sep='\t', names=header_names)
    #     total_data = [test_data]

    if data == 'test':
        test_data = pd.read_table(file_path + 'final_test.txt', sep='\t', names=header_names)
        total_data = [test_data]

    if data == 'test_binirized':
        test_data = pd.read_table(file_path + 'test_binirized.tsv', sep='\t', names=header_names)
        total_data = [test_data]

    if data == 'train_valid':
        train_data = pd.read_table(file_path + 'train.tsv', sep='\t', names=header_names)
        dev_data = pd.read_table(file_path + 'valid.tsv', sep='\t', names=header_names)
        total_data = [train_data, dev_data]

    if data == 'train_valid_test':
        train_data = pd.read_table(file_path + 'train.tsv', sep='\t', names=header_names)
        dev_data = pd.read_table(file_path + 'valid.tsv', sep='\t', names=header_names)
        test_data = pd.read_table(file_path + 'test.tsv', sep='\t', names=header_names)
        total_data = [train_data, dev_data, test_data]

    total_data = pd.concat(total_data)
    return total_data
