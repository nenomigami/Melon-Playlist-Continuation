# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:32:46 2020

@author: ghdbs
"""

from implicit.als import AlternatingLeastSquares as ALS

class Wrmf:
    
    def __init__(self):
        params = params.copy()
        self.c = params["c"]
        del params["c"]
        self.model = ALS(**params)

    def fit(self, X, y):
        self.model.fit(self.c * X.T)
        
        pass
    
    def predict(self, idx):
        isinstance(idx, )
        pass
    def recommend(self):
        pass
    
    