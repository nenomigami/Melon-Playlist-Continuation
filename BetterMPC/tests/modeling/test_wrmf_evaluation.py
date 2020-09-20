# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:41:19 2020

@author: ghdbs
"""

import pandas as pd
import numpy as np
import stage1_config
from src.modeling.wrmf import Wrmf
from src.modeling.wrmf_evaluation import recommend, calculate_recall, _recall
from pandas.testing import *
from numpy.testing import *
import dill as pkl

class TestWrmfEvaluation(object):
    
    def test_recommend_on_sample(self):
        with open(stage1_config.SAVE_FOLDER + "model1.pkl","rb") as f:
            model = pkl.load(f)
        with open(stage1_config.STAGE1_PROCESSED_TRAIN_FILE,"rb") as f:
            train = pkl.load(f)
        with open(stage1_config.SAVE_FOLDER + "pipeline1.pkl", "rb") as f:
            pipeline = pkl.load(f)
        test = pd.concat(
                    [pd.read_json(f) for f in stage1_config.STAGE1_VALID_Q]
                    ).reset_index(drop = True)
        idx = np.arange(train.shape[0]-len(test),train.shape[0])[0:100]
        pred = model.predict(idx,200,20)
        test_argument = recommend(pred, pipeline)
        query = pd.concat(
                [pd.read_json(f) for f in stage1_config.STAGE1_VALID_Q]
                ).reset_index(drop = True).iloc[0:100]
        test = pd.concat(
                [pd.read_json(f) for f in stage1_config.STAGE1_VALID_A]
                ).reset_index(drop = True).iloc[0:100]
        calculate_recall(query, test_argument, test)


