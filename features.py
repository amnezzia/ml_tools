


import pandas as pd
import numpy as np
import scipy.sparse as sp

from general_utils import load_object, save_object

class FeatureMaker(object):
    '''
    base feature maker
    '''

    def __init__(self, in_path=None, out_path=None):

        self.in_path = in_path
        self.out_path = out_path

        if in_path is not None:
            self._read_data()

    def _read_data(self):

        if self.in_path.endswith('.csv'):
            self._read_method = pd.read_csv
        elif self.in_path.endswith('.pickle'):
            self._read_method = load_object
        else:
            print("Not sure what to do with this file...")

        self.in_data = self._read_method(self.in_path)

        self.is_sparse = False
        if isinstance(self.in_data, sp.spmatrix):
            self.is_sparse = True

        self.num_rows = self.in_data.shape[0]

        self.in_columns = getattr(self.in_data, 'columns', [])

    def fit(self):
        '''
        fit feature maker using input data
        :return:
        '''
        pass

    def make_features(self, fpath=None):
        '''
        by default will make features from input data
        optionally can make features from data in fpath
        :param fpath: optionally provide additional data for making features
        :return: None
        '''
        df = self.in_data
        if fpath is not None:
            df = pd.read_csv(fpath)
            for c in self.in_columns:
                assert (c in df.columns)

        self.features = self._make_feat(df)


    def _make_feat(self, df):
        '''
        the actual method for feature generation
        :param df: in data
        :return: out data
        '''
        # create one random column
        feat_df = df[['_id',]].copy()
        feat_df['random'] = np.random.randn(df.shape[0], 1)

        return feat_df


    def save_features(self, fpath=None):
        '''
        save created features into a file
        :param fpath: path to save into
        :return:
        '''
        if fpath is None and self.out_path is None:
            raise Exception("Need a path to save features")
        elif fpath is not None:
            # override
            self.out_path = fpath

        with open(fpath, 'wb') as f:
            pickle.dump(self.features, f)
        #self.features.to_csv(fpath, index=False)
        print("Saved features to ", fpath)

    def save(self, fpath):
        '''
        save itself
        :param fpath:
        :return:
        '''
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fpath):
        '''
        load pickled class instance
        :param fpath:
        :return:
        '''
        with open(fpath, 'rb') as f:
            tmp = pickle.load(f)

        return tmp
