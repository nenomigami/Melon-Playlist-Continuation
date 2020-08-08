# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 01:12:46 2020

@author: ghdbs
"""

import stage1_config
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import CategoricalDtype

# categorical missing value imputer
class Uir_matrix_maker(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.len_features = {}
    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        df_list = []
        for feature in self.variables:
            df = X[["id",feature]].explode(feature)
            df.columns = ["id", "songs_tags"]
            df = df.dropna()
            self.len_features[feature] = len(df)
            df_list.append(df)
        uir_matrix = pd.concat(df_list).reset_index(drop = True)
        return uir_matrix

class Csr_matrix_maker(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, uir_matrix):
        uir_matrix = uir_matrix.copy()
        id_u = list(uir_matrix["id"].unique())
        songs_tags_u = list(uir_matrix["songs_tags"].unique())
        id_c = CategoricalDtype(id_u, ordered=False)
        songs_tags_c = CategoricalDtype(songs_tags_u, ordered=False)
        
        data = np.ones(len(uir_matrix))
        row = uir_matrix["id"].astype(id_c).cat.codes
        col = uir_matrix["songs_tags"].astype(songs_tags_c).cat.codes
        sparse_matrix = csr_matrix((data, (row, col)),
                                   shape=(len(id_u), len(songs_tags_u)))
        return sparse_matrix   
if __name__ == "__main__":   
    uir_matrix_maker = Uir_matrix_maker(stage1_config.TARGET_VARS)
    test_argument = pd.read_json(stage1_config.TRAINING_DATA_FILE)
    test_argument = uir_matrix_maker.fit_transform(test_argument)
    actual = sum(Csr_matrix_maker().fit_transform(test_argument).data)
    expected = sum(uir_matrix_maker.len_features.values())
    assert test_argument.isna().sum().sum() == 0 
    assert actual == expected
    pass