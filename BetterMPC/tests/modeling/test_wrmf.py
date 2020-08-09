# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:33:28 2020

@author: ghdbs
"""
import pytest
import dill as pkl
import pandas as pd
import stage1_config
from src.modeling import wrmf
class TestWrmf(object):
    
    def test_train_on_samples_and_predict_one_row(self):
        with open(STAGE1_PROCESSED_TRAIN_FILE,"rb") as f:
            train = pkl.load(f)
        test_argument = 
        
        pass
