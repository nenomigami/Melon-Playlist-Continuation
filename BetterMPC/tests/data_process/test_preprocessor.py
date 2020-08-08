# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 01:12:46 2020

@author: ghdbs
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import stage1_config
from src.data_process.preprocessor import Uir_matrix_maker, Csr_matrix_maker
from pandas.testing import *
from numpy.testing import *
class TestUirMatrixMaker(object):
    
    def test_on_one_row(self):
        
        
        test_argument = pd.DataFrame(
                                {'tags': [['락']],
                                 'id': [61281],
                                 'plylst_title': ['여행같은 음악'],
                                 'songs': [[525514, 129701, 383374, 562083, 297861,
                                            139541, 351214, 650298, 531057]],
                                 'like_cnt': [71],
                                 'updt_date': ['2013-12-19 18:36:19.000']}
                                )
        
        expected = pd.DataFrame(
                                {'id': [61281,61281,61281,61281,61281,
                                        61281,61281,61281,61281,61281],
                                 'songs_tags': [525514, 129701, 383374, 562083, 297861,
                                            139541, 351214, 650298, 531057, "락"]}
                                )
        
        actual = Uir_matrix_maker(stage1_config.TARGET_VARS).fit_transform(test_argument)                       
        assert_frame_equal(actual, expected)
        
    def test_length_on_train_sample(self):
        train = pd.read_json(stage1_config.TRAINING_DATA_FILE)
        test_argument = train.iloc[:20]
        
        expected = 863
        
        actual = len(Uir_matrix_maker(stage1_config.TARGET_VARS).fit_transform(test_argument))
            
        assert(actual == expected)
        
    def test_length_on_whole_stage1_train(self):
        test_argument = pd.concat(
                        [pd.read_json(f) for f in stage1_config.STAGE1_TRAIN]
                        ).reset_index(drop = True)
        expected = 5965364
        
        actual = len(Uir_matrix_maker(stage1_config.TARGET_VARS).fit_transform(test_argument))
            
        assert(actual == expected)

class TestCsrMatrixMaker(object):
    #pandas category 하니까 저절로 sorting이 된다 유의
    def test_on_small_dataset(self):
        test_argument = pd.DataFrame({"id" : [1,1,1,2,3,3],
                                      "songs_tags" : ["10","20","30","10","20","락"]})
        expected = csr_matrix((np.ones(6),
                            ([0,0,0,1,2,2],
                             [0,1,2,0,1,3])),
                            shape = [3,4])    
        actual = Csr_matrix_maker().fit_transform(test_argument)
        assert_array_equal(actual.data, expected.data)
        assert_array_equal(actual.shape, expected.shape)
        
    def test_length_on_whole_stage1_train(self):
        uir_matrix_maker = Uir_matrix_maker(stage1_config.TARGET_VARS)
        test_argument = pd.concat(
                        [pd.read_json(f) for f in stage1_config.STAGE1_TRAIN]
                        ).reset_index(drop = True)
        test_argument = uir_matrix_maker.fit_transform(test_argument)
        actual = sum(Csr_matrix_maker().fit_transform(test_argument).data)
        expected = sum(uir_matrix_maker.len_features.values())
        assert actual == expected
        
    def test_length_on_train(self):
        uir_matrix_maker = Uir_matrix_maker(stage1_config.TARGET_VARS)
        test_argument = pd.read_json(stage1_config.TRAINING_DATA_FILE)
        test_argument = uir_matrix_maker.fit_transform(test_argument)
        actual = sum(Csr_matrix_maker().fit_transform(test_argument).data)
        expected = sum(uir_matrix_maker.len_features.values())
        assert test_argument.isna().sum().sum() == 0 
        assert actual == expected