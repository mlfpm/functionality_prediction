# @author semese

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

pd.options.mode.chained_assignment = None


class ObjectDict(dict):
    '''
    Interface similar to an argparser.
    '''

    def __init__(self):
        super().__init__()

    def __setattr__(self, attr, value):
        self[attr] = value
        return self[attr]

    def __getattr__(self, attr):
        if attr.startswith('_'):
            # https://stackoverflow.com/questions/10364332/how-to-pickle-python-object-derived-from-dict
            raise AttributeError
        return dict(self)[attr]

    @property
    def __dict__(self):
        return dict(self)


class TimeseriesDataset(Dataset):
    '''
    Custom Dataset subclass.
    Serves as input to DataLoader to impute time series sequences.
    DataLoader using this dataset will output batches of `(batch_size, seq_len, day_len, n_features)` shape.
    Suitable as an input to the pipeline.
    '''

    def __init__(self, X: list, y: list, hparams: ObjectDict):
        '''
        Class initialiser.
        '''
        self.X = X
        self.y = y
        self.hmm = pickle.load(open(hparams.hmm_path, 'rb'))
        self.add_posteriors = hparams.add_posteriors
        self.only_posteriors = hparams.only_posteriors

    def __len__(self):
        return self.y.__len__()

    def __getitem__(self, index):
        '''
        Get item at ´index´ from X and y for data loader.
        '''
        x_t_i = self.impute_missing_data_from_hmm(self.X[index])

        return torch.tensor(x_t_i).float(), torch.tensor(self.y[index]).float()

    def impute_missing_data_from_hmm(self, obs_seqs):
        '''
        Function to perform data imputation on the observation sequences.
        For this a pre-trained HMM model is used. We decode the observation sequences
        with the HMM using the Viterbi algorithm, then for each time step, we draw
        samples from the corresponding state and fill in the missing values from the sample.
        As an extra feature we add the posterior probabilities of the imputed
        observation sequence.
        '''
        feature_seqs = []
        # print('Obs.sequence: {}'.format(obs_seqs.shape))
        for obs_seq in obs_seqs:
            # print('Obs.sequence: {}'.format(obs_seq.shape))
            # decode observation sequence
            state_seq = self.hmm.decode([obs_seq], algorithm='viterbi')[1][0]
            # print('State sequence: {}'.format(state_seq))
            imp_seq = obs_seq.copy()
            for t, obs in enumerate(imp_seq):
                # create mask for NaN values in the observation
                nan_mask = np.isnan(obs)
                if np.any(nan_mask):
                    # generate a sample from the state and make sure they are all positive
                    sample = self.hmm.generate_sample_from_state(state_seq[t])
                    # print('Sample {}'.format(sample))
                    # print('Sample[nan_mask] {}'.format(sample[nan_mask]))
                    # fill in missing values from the generated sample
                    #print('masked_seq[t] {}'.format(masked_seq[t]))
                    # print('masked_seq[t][nan_mask] {}'.format(
                    #    masked_seq[t][nan_mask]))
                    imp_seq[t][nan_mask] = sample[nan_mask]

            if self.add_posteriors:
                # compute the posteriors for the sequence
                posteriors = self.hmm.score_samples([obs_seq])[0]
                if not self.only_posteriors:
                    # append the posteriors to the features
                    feature_seq = np.hstack((imp_seq, posteriors)).astype(
                        np.float64
                    )
                else:
                    feature_seq = posteriors.astype(np.float64)
            else:
                feature_seq = imp_seq.astype(np.float64)
                # print(feature_seq)
            feature_seqs.append(feature_seq)

        return feature_seqs


class MixDataset(TimeseriesDataset):
    '''
    Custom Dataset subclass. Serves as input to DataLoader to impute the time series
    sequences and add the demographic data. DataLoader using this dataset will output
    batches of tuples of shape ([batch_size, seq_len, day_len, n_temp_features], [batch_size, n_demogr_features]).
    Suitable as an input to the MobilityClassifier.
    '''

    def __init__(self, X: tuple, y: list, hparams: ObjectDict):
        '''
        Class initialiser.

        :param X: tuple containing a list for the temporal data and one for the socio-demographic data
        :type X: tuple
        :param y: target outcomes
        :type y: list
        :param hparams: hyperparameters 
        :type hparams: ObjectDict
        '''
        super().__init__(X[0], y, hparams)

        self.X_demgr = X[1]

    def __getitem__(self, index: int):
        '''
        Get item from X_temp, X_demogr and y for data loader.

        :param index: index of the item to get
        :type index: int
        :return: (observation sequence, demographic data) tuple and label pair at 'index'
        :rtype: tuple
        '''

        x_t_i, y_i = super().__getitem__(index)

        return tuple((x_t_i, torch.tensor(self.X_demgr[index]).float())), y_i


class MobilityDataModule:
    '''
    Serves the purpose of aggregating all data loading and processing work in one place.
    '''

    def __init__(self, hparams: dict):
        '''
        Class initialiser.
        '''
        super().__init__()

        self.hparams = ObjectDict()
        self.hparams.update(
            hparams.__dict__ if hasattr(hparams, '__dict__') else hparams
        )

        self.setup()

    def setup(self):
        '''
        Load training data and create train-validation-test splits.
        '''
        # load the data
        df = pd.read_csv(self.hparams.data_path)
        if self.hparams.with_demogr:
            df.dropna(subset=self.hparams.x_d_cols, how='any')

        # make sure the temporal values are numerical
        self.x_t_cols = np.asarray(
            list(self.hparams.x_t_cols.values())).ravel()
        df.loc[:, self.x_t_cols] = df[self.x_t_cols].apply(
            pd.to_numeric, errors='coerce')

        if self.hparams.test_set_only:
            # don't split the dataset
            self.X_test, self.y_test = self.create_dataset(df, train=True)
            if self.hparams.verbose:
                self.print_dataset_info(self.X_test, 'Test set')
        else:
            if self.hparams.random_split:
                # random-permute patient ids and select train-valid-test groups
                train_ids, val_ids, test_ids = self.split_patients(df)
            else:
                # used pre-defined split
                train_ids, val_ids, test_ids = self.hparams.train_test_split

            # get train-valid-test sets
            self.X_train, self.y_train = self.create_dataset(
                df[df.user.isin(train_ids)], train=True)
            self.X_val, self.y_val = self.create_dataset(
                df[df.user.isin(val_ids)])
            self.X_test, self.y_test = self.create_dataset(
                df[df.user.isin(test_ids)])

            if self.hparams.verbose:
                self.print_dataset_info(
                    self.X_train, 'Training set')
                self.print_dataset_info(
                    self.X_val, 'Validation set')
                self.print_dataset_info(self.X_test, 'Test set')

    def split_patients(self, df):
        '''
        Train-validation-test split by patients.

        :param df: dataframe with the data and id column to split by
        :type df: pd.DataFrame
        :return: arrays with patient ids for the datasets
        :rtype: tuple
        '''
        np.random.seed(1)
        if self.hparams.use_all:
            patient_ids = np.random.permutation(df.user.unique())
        else:
            patient_ids = np.random.permutation(
                df.user.unique())[:int(df.user.nunique()/100)]

        train_ids = patient_ids[:int(
            self.hparams.train_split * len(patient_ids))]
        remaining_ids = patient_ids[int(
            self.hparams.train_split * len(patient_ids)):]
        val_ids, test_ids = remaining_ids[:int(
            0.5 * len(remaining_ids))], remaining_ids[int(0.5 * len(remaining_ids)):]

        return train_ids, val_ids, test_ids

    def create_dataset(self, df: pd.DataFrame, train: bool = False):
        '''
        Standardise the data and split up the patient sequences and extract labels.
        '''

        # log-transform + normalise the variables
        df.loc[:, self.x_t_cols] = np.log(df[self.x_t_cols].values+1)

        # if training set, create the scaler otherwise just scale
        if train:
            self.scaler = StandardScaler().fit(
                df[self.x_t_cols].values)

        df.loc[:, self.x_t_cols] = self.scaler.transform(
            df[self.x_t_cols].values)

        if self.hparams.autoregression:
            return self.create_autoregression_dataset(df)

        return self.create_labelled_dataset(df)

    def create_autoregression_dataset(self, df):
        '''
        Extract 30-days of data and as label the average of the next 30-days.
        '''
        X, y = [], []
        for _, df_p in df.groupby('user'):
            L = len(df_p)
            tw = self.hparams.seq_len
            for i in range(0, L-tw, tw):
                data = df_p.iloc[i:i+tw, :]
                label = df_p.iloc[i+tw:i+tw+1, :]

                if data[self.x_t_cols].isnull().values.all() or \
                        any(label[cols].isnull().values.all() for cols in self.hparams.x_t_cols.values()):
                    continue

                X.append(np.asarray([
                    np.hstack([
                        x_t[cols].values.reshape(-1, 1)
                        for cols in self.hparams.x_t_cols.values()
                    ])
                    for _, x_t in data.iterrows()
                ]).astype(np.float64))

                y.append(np.asarray([
                    np.nanmean(
                        np.nansum(label[cols].values, axis=1), axis=None)
                    for cols in self.hparams.x_t_cols.values()
                ]).astype(np.float64))

        assert len(X) == len(y)

        return X, y

    def create_labelled_dataset(self, df):
        '''
        Extract 30-days sequences if they have a corresponding label. 
        '''
        X, X_demgr, y = [], [], []
        for _, df_p in df.groupby('user'):
            # extract indices of target label
            label_idx = [
                idx[0] for idx in np.argwhere(df_p.loc[:, self.hparams.y_col].notnull().values).tolist()
            ]
            # for each index
            for idx in label_idx:
                dont_save = False
                if 10 < len(df_p) <= self.hparams.seq_len:
                    if df_p[self.x_t_cols].isnull().sum().sum() < 0.5*np.prod(df_p[self.x_t_cols].shape):
                        dont_save = True
                    else:
                        padding = np.empty(
                            (self.hparams.seq_len-len(df_p), df_p.shape[1]))
                        padding[:] = np.NaN
                        data = pd.DataFrame(
                            data=np.vstack([padding, df_p.values]),
                            columns=df_p.columns
                        )
                else:
                    tw = self.hparams.seq_len
                    l_idx = 0 if (idx + 1) - \
                        (tw+10) < 0 else (idx + 1) - (tw+10)
                    r_idx = len(df_p) if (idx + 1) + \
                        (tw+10) > len(df_p) else (idx + 1) + (tw+10)
                    min_nan_cnt, min_nan_seq = np.inf, None
                    for i in range(l_idx, r_idx-tw):
                        data = df_p.iloc[i:i+tw, :]
                        if data[self.x_t_cols].isnull().sum().sum() < min_nan_cnt:
                            min_nan_cnt = data[self.x_t_cols].isnull(
                            ).sum().sum()
                            min_nan_seq = data
                    if min_nan_seq is None:
                        dont_save = True
                    else:
                        data = min_nan_seq
                if not dont_save:
                    X.append(np.asarray([
                        np.hstack([
                            x_t[cols].values.reshape(-1, 1)
                            for cols in self.hparams.x_t_cols.values()
                        ])
                        for _, x_t in data.iterrows()
                    ]).astype(np.float64))

                    if self.hparams.with_demogr:
                        X_demgr.append(
                            df_p.loc[:, self.hparams.x_d_cols].values[idx, :].astype(np.float64))
                    y.append(df_p.loc[:, self.hparams.y_col].values[idx])

        if self.hparams.with_demogr:
            assert len(X) == len(X_demgr)

        assert len(X) == len(y)

        if self.hparams.regression:
            y /= np.amax(y)

        if self.hparams.with_demogr:
            return (X, X_demgr), y

        return X, y

    def print_dataset_info(self, X, which):
        '''
        Print dataset shape. 
        '''
        print(which)
        if self.hparams.with_demogr:
            print(len(X[0]), len(X[0][0]), X[0][0][0].shape)
            print(len(X[1]), X[1][0].shape)
        else:
            print(len(X), len(X[0]), X[0][0].shape)

    def get_sampler(self, y):
        '''
        Created weighted random sampler for the dataloader, if the output in binary or categorical.
        '''
        # class weighting
        _, counts = np.unique(y, return_counts=True)
        class_weights = [sum(counts)/c for c in counts]

        # assign weight to each input sample
        example_weights = [class_weights[int(y_i)] for y_i in y]

        return WeightedRandomSampler(example_weights, len(y))

    def train_dataloader(self):
        '''
        Getter for train loader.
        '''
        if self.hparams.with_demogr:
            train_dataset = MixDataset(
                self.X_train, self.y_train, self.hparams)
        else:
            train_dataset = TimeseriesDataset(
                self.X_train, self.y_train, self.hparams)

        if self.hparams.regression:
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.num_workers,
                                      drop_last=True)
        else:
            train_loader = DataLoader(train_dataset,
                                      sampler=self.get_sampler(self.y_train),
                                      batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.num_workers,
                                      drop_last=True)

        return train_loader

    def val_dataloader(self):
        '''
        Getter for validation loader. 
        '''
        if self.hparams.with_demogr:
            val_dataset = MixDataset(
                self.X_val, self.y_val, self.hparams)
        else:
            val_dataset = TimeseriesDataset(self.X_val,
                                            self.y_val,
                                            self.hparams)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.hparams.batch_size if len(
                                    self.y_val) > self.hparams.batch_size else len(self.y_val),
                                num_workers=self.hparams.num_workers, drop_last=True)

        return val_loader

    def test_dataloader(self):
        '''
        Getter for test loader
        '''
        if self.hparams.with_demogr:
            test_dataset = MixDataset(
                self.X_test, self.y_test, self.hparams)
        else:
            test_dataset = TimeseriesDataset(self.X_test,
                                             self.y_test,
                                             self.hparams)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.hparams.batch_size if len(
                                     self.y_test) > self.hparams.batch_size else len(self.y_test),
                                 num_workers=self.hparams.num_workers, drop_last=True)

        return test_loader
