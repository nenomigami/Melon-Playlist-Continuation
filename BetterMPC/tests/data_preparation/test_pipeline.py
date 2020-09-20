# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 00:44:38 2020

@author: ghdbs
"""

from src.data_process import pipeline
import stage1_config
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import *

class TestPipeLine(object):
    
    def test_pipeline_on_one_line_train(self):
        test_argument = pd.DataFrame(
                                {'tags': [['락']],
                                 'id': [61281],
                                 'plylst_title': ['여행같은 음악'],
                                 'songs': [[525514, 129701, 383374, 562083, 297861,
                                            139541, 351214, 650298, 531057]],
                                 'like_cnt': [71],
                                 'updt_date': ['2013-12-19 18:36:19.000']}
                                )
        
        expected = csr_matrix(([1,1,1,1,1,1,1,1,1,1],
                              ([0,0,0,0,0,0,0,0,0,0],
                              [0,1,2,3,4,5,6,7,8,9])))
        actual = pipeline.wrmf_data_pipe.fit_transform(test_argument)
        assert_array_equal(actual.data, expected.data)
        assert_array_equal(actual.indptr, expected.indptr)
        assert_array_equal(actual.indices, expected.indices)

    def test_pipeline_on_two_line_train(self):
        test_argument = pd.DataFrame(
                                {'tags': [['a'], ['b']],
                                 'id': [2,3],
                                 'plylst_title': ['여행같은 음악',"제목1"],
                                 'songs': [[1, 2, 3],
                                           [4, 5, 6]],
                                 'like_cnt': [71,21],
                                 'updt_date': ['2013-12-19 18:36:19.000',
                                               '2013-12-19 18:36:19.000']}
                                )
        
        
        expected = csr_matrix((np.ones(8),
                              ([0,0,0,1,
                                1,1,0,1,],
                              [0,1,2,3,
                               4,5,6,7])))
        actual = pipeline.wrmf_data_pipe.fit_transform(test_argument)
        assert_array_equal(actual.data, expected.data)
        assert_array_equal(actual.indptr, expected.indptr)
        assert_array_equal(actual.indices, expected.indices)
        
