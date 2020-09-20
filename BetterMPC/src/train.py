# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 19:01:31 2020

@author: ghdbs
"""

from sklearn.pipeline import Pipeline
from src.data_process import preprocessor as pp
from src.data_process.pipeline import wrmf_data_pipe
import stage1_config
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import *
import dill as pkl

Datagenerator()


stage1_train_raw = pd.concat(
                [pd.read_json(f) for f in stage1_config.STAGE1_TRAIN]
                ).reset_index(drop = True)
stage1_valid_raw = pd.concat(
                [pd.read_json(f) for f in stage1_config.STAGE1_VALID_A]
                ).reset_index(drop = True)
processed_train = wrmf_data_pipe.fit_transform(stage1_train)

with open(stage1_config.SAVE_FOLDER + "pipeline1.pkl", "rb") as f:
    pkl.dump(wrmf_data_pipe, f)
with open(stage1_config.STAGE1_PROCESSED_TRAIN_FILE, "rb") as f:
    stage1_train_processed = pkl.load(f)
    
with open(stage1_config.SAVE_FOLDER + "model1.pkl","rb") as f:
    model = pkl.load(f)
    
with open(stage1_config.STAGE2_TRAIN, "rb") as f:
    stage2_train = pkl.load(f)

stage1_valid_raw = pd.concat(
                [pd.read_json(f) for f in stage1_config.STAGE1_VALID_A]
                ).reset_index(drop = True)  
stage2_test1_raw = pd.read_json(stage1_config.VALIDATING_DATA_FILE)
stage2_test2_raw = pd.read_json(stage1_config.TESTING_DATA_FILE)


len(stage2_test1_raw) #23015
len(stage2_test2_raw) #10740
len(stage1_valid_raw) #13958
147936 - (23015 + 10740 + 13958)
stage2_test1 = stage1_train_processed[-47713 : -24698]

stage2_test1_idx = np.arange(100223, 100223+23015)
stage2_test1 = model.predict(stage2_test1_idx,1000,100)

with open(stage1_config.STAGE2_TEST1, "wb") as f:
    pkl.dump(f, stage2_test1)

stage2_test2_idx = np.arange(100223+23015, 100223+23015+10740)
stage2_test2 = model.predict(stage2_test1_idx,1000,100)

with open(stage1_config.STAGE2_TEST2, "wb") as f:
    pkl.dump(f, stage2_test2)

#개선사항 : stage1은 일종의 transformer 이기때문에 새로운 래퍼를 만들어서 transform 시키는 게 있으면 좋겠음
stage2_split
evaluating 만들기
song_meta 합치기
stage2 파이프라인 만들기
catboost 적합하기


