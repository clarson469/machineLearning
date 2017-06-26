import os, settings
import pandas as pd, numpy as np
from .data import read_txt

class CensusWage(object):
    def __init__(self):
        fname = os.path.join(settings.STATS_DIR, 'census-wage.txt')
        col_fname = os.path.join(settings.STATS_DIR, 'census-wage-info.txt')
        names = read_txt(col_fname)
        data_raw = pd.read_csv(fname, **{
            'header':None,
            'names':names,
            'sep':',\s',
            'na_values':['?'],
            'engine':'python'
        }).dropna()
        y_raw = data_raw['income']
        X_raw = data_raw[names[:-1]]

        X = self.dummy_data(X_raw)
        X_norm = self.normalise(X)
        y = y_raw.replace(['<=50K', '>50K'], [0, 1])

        self.split_data(X_norm, y)
        self.input_dim = self.Xtr.shape[1]
        self.output_dim = 1


    def dummy_data(self, data):
        o_cols = []
        for column in data:
            if data[column].dtype != 'O':
                continue
            o_cols.append(column)
            dummy = pd.get_dummies(data[column], prefix=column)
            for d_column in dummy.columns:
                data = data.join(dummy[d_column])

        return data.drop(o_cols, axis=1)

    def normalise(self, data):
        bin_cols = [col for col in data if data[[col]].isin([0,1]).all().values]

        sig = data.mean(axis=0)
        lmb = data.std(axis=0)
        for col in bin_cols:
            sig[col] = 0
            lmb[col] = 1
        return (data - sig) / lmb

    def split_data(self, X, y):
        i = np.floor(X.shape[0] / 7).astype('int')
        
        self.Xtr = X[i:].as_matrix()
        self.Ytr = y[i:].as_matrix()
        self.Ytr = self.Ytr.reshape(self.Ytr.shape[0], 1)

        self.Xcv = X[:i].as_matrix()
        self.Ycv = y[:i].as_matrix()
        self.Ycv = self.Ycv.reshape(self.Ycv.shape[0], 1)
