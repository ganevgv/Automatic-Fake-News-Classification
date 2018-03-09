class majority(object):
    def __init__(self, new_speakers_label_arg=None):
        self.new_speakers_label_arg = new_speakers_label_arg
        self.new_speakers_label = None
        self.speakers_credit = None

    def fit(self, df):
        self.new_speakers_label = df['label'].value_counts().idxmax()
        # print(df['label'].value_counts().idxmax())

        label_list = df['label'].unique()

        df_reduced = df.drop_duplicates(subset=['speaker'], keep='first')
        df_reduced = df_reduced[['speaker', 'barely-true-counts', 'false-counts', 'half-true-counts', 'mostly-true-counts', 'pants-fire-counts']]

        for label in label_list:
            label_data = df[df['label'] == label]
            labels_counts = label_data.groupby('speaker').agg({'label': len})
            labels_counts_dic = labels_counts['label']
            df_reduced[label] = df_reduced['speaker'].map(labels_counts_dic)
            df_reduced[label].fillna(0, inplace=True)

        df_reduced = df_reduced.set_index('speaker', drop=True)
        df_reduced = df_reduced.iloc[:, 5:]
        # print(df_reduced.loc['barack-obama'])
        df_reduced['majority'] = df_reduced.idxmax(axis=1)

        self.speakers_credit = df_reduced

    def predict(self, df):
        n_speakers = len(df['speaker'].unique())
        new_speakers = 0

        for index, row in df.iterrows():
            try:
                df.set_value(index, 'majority', self.speakers_credit.loc[row['speaker']]['majority'])
            except KeyError:
                new_speakers += 1
                if self.new_speakers_label_arg is None:
                    df.set_value(index, 'majority', self.new_speakers_label)
                else:
                    df.set_value(index, 'majority', self.new_speakers_label_arg)
                continue

        if len(list(df['label'].unique())) == 2:
            df['majority'] = df['majority'].astype('bool')

        predicted_lables = df['majority']
        print('no train data for: {} speakers out of {} in this data'.format(new_speakers, n_speakers))

        return predicted_lables
