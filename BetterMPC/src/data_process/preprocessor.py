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
        self.nunique_features = {}
    def fit(self, X, y=None):
        X = X.copy()
        for feature in self.variables:
            df = X[["id",feature]].explode(feature)
            self.nunique_features[feature] = df[feature].nunique()
            df.columns = ["id", "songs_tags"]
            df = df.dropna()

        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        df_list = []
        for feature in self.variables:
            df = X[["id",feature]].explode(feature)
            df.columns = ["id", "songs_tags"]
            df = df.dropna()
            df_list.append(df)
        uir_matrix = pd.concat(df_list).reset_index(drop = True)
        return uir_matrix

class Csr_matrix_maker(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.id_c = None
        self.songs_tags_c =None
        
    def fit(self, uir_matrix, y=None):
        uir_matrix = uir_matrix.copy()
        id_u = list(uir_matrix["id"].unique())
        songs_tags_u = list(uir_matrix["songs_tags"].unique())
        self.id_c = CategoricalDtype(id_u, ordered=False)
        self.songs_tags_c = CategoricalDtype(songs_tags_u, ordered=False)
        return self

    def transform(self, uir_matrix):
        data = np.ones(len(uir_matrix))
        row = uir_matrix["id"].astype(self.id_c).cat.codes
        col = uir_matrix["songs_tags"].astype(self.songs_tags_c).cat.codes
        sparse_matrix = csr_matrix((data, (row, col)),
                                   shape=(len(self.id_c.categories),
                                          len(self.songs_tags_c.categories)))
        return sparse_matrix   
if __name__ == "__main__":   
    uir_matrix_maker = Uir_matrix_maker(stage1_config.TARGET_VARS)
    test_argument = pd.read_json(stage1_config.TRAINING_DATA_FILE)
    test_argument = uir_matrix_maker.fit_transform(test_argument)
    csr_matrix_maker = Csr_matrix_maker()
    actual = csr_matrix_maker.fit_transform(test_argument)
    csr_matrix_maker.songs_tags_c.categories[615141]
    assert test_argument.isna().sum().sum() == 0 
    assert actual == expected
    pass