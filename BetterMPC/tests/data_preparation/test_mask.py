# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 19:49:50 2020

@author: ghdbs
"""
import pytest
import stage1_config
import pandas as pd
from pandas.testing import *
from src.data_preparation.DataGenerator import DataGenerator
#with open("test.txt", "a") as f:
#    f.write(str(pd.read_json(stage1_config.TRAINING_DATA_FILE).head(5).to_dict("list")))
class TestMask(object):
    
    def test_song_only_case_on_one_row(self):
         test_argument = pd.DataFrame(
                                        {'tags': [['락']],
                                         'id': [61281],
                                         'plylst_title': ['여행같은 음악'],
                                         'songs': [[525514, 129701, 383374, 562083, 297861,
                                                    139541, 351214, 650298, 531057, 205238,
                                                    706183, 127099, 660493, 461973, 121455,
                                                    72552, 223955, 324992, 50104]],
                                         'like_cnt': [71],
                                         'updt_date': ['2013-12-19 18:36:19.000']}
                                        )
         expected = pd.DataFrame(
                                        {'tags': [[]],
                                         'id': [61281],
                                         'plylst_title': ['여행같은 음악'],
                                         'songs': [[525514, 129701, 383374, 562083, 297861,
                                                    139541, 351214, 650298, 531057]],
                                         'like_cnt': [71],
                                         'updt_date': ['2013-12-19 18:36:19.000']}
                                        )
         
         actual = DataGenerator()._mask(test_argument,["songs"], ["tags"])
         assert_series_equal(expected["tags"], actual[0]["tags"])
         assert len(expected["songs"]) == len(actual[0]["songs"])

    def test_tag_only_case_on_one_row(self):
        test_argument = pd.DataFrame(
                                   {'tags': [['락']],
                                    'id': [61281],
                                    'plylst_title': ['여행같은 음악'],
                                    'songs': [[525514, 129701, 383374, 562083, 297861,
                                               139541, 351214, 650298, 531057, 205238,
                                               706183, 127099, 660493, 461973, 121455,
                                               72552, 223955, 324992, 50104]],
                                    'like_cnt': [71],
                                    'updt_date': ['2013-12-19 18:36:19.000']}
                                   )
        expected = pd.DataFrame(
                                   {'tags': [[]],
                                    'id': [61281],
                                    'plylst_title': ['여행같은 음악'],
                                    'songs': [[]],
                                    'like_cnt': [71],
                                    'updt_date': ['2013-12-19 18:36:19.000']}
                                   )
        actual = DataGenerator()._mask(test_argument,["tags"],["songs"])
        assert_series_equal(expected["tags"], actual[0]["tags"])
        assert len(expected["songs"]) == len(actual[0]["songs"])
        
    def test_song_tag_case_on_one_row(self):
        test_argument = pd.DataFrame(
                                       {'tags': [['락',"리리링"]],
                                        'id': [61281],
                                        'plylst_title': ['여행같은 음악'],
                                        'songs': [[525514, 129701, 383374, 562083, 297861,
                                                   139541, 351214, 650298, 531057, 205238,
                                                   706183, 127099, 660493, 461973, 121455,
                                                   72552, 223955, 324992, 50104]],
                                        'like_cnt': [71],
                                        'updt_date': ['2013-12-19 18:36:19.000']}
                                       )
        expected = pd.DataFrame(
                                       {'tags': [['락']],
                                        'id': [61281],
                                        'plylst_title': ['여행같은 음악'],
                                        'songs': [[]],
                                        'like_cnt': [71],
                                        'updt_date': ['2013-12-19 18:36:19.000']}
                                       )
        actual = DataGenerator()._mask(test_argument,["songs","tags"],[])
        assert len(expected["tags"]) == len(actual[0]["tags"])
        assert len(expected["songs"]) == len(actual[0]["songs"])
    def test_title_only_case_on_one_row(self):
        test_argument = pd.DataFrame(
                                       {'tags': [['락',"리리링"]],
                                        'id': [61281],
                                        'plylst_title': ['여행같은 음악'],
                                        'songs': [[525514, 129701, 383374, 562083, 297861,
                                                   139541, 351214, 650298, 531057, 205238,
                                                   706183, 127099, 660493, 461973, 121455,
                                                   72552, 223955, 324992, 50104]],
                                        'like_cnt': [71],
                                        'updt_date': ['2013-12-19 18:36:19.000']}
                                       )
        expected = pd.DataFrame(
                                       {'tags': [[]],
                                        'id': [61281],
                                        'plylst_title': ['여행같은 음악'],
                                        'songs': [[]],
                                        'like_cnt': [71],
                                        'updt_date': ['2013-12-19 18:36:19.000']}
                                       )
        actual = DataGenerator()._mask(test_argument,[],["songs","tags"])
        assert len(expected["tags"]) == len(actual[0]["tags"])
        assert len(expected["songs"]) == len(actual[0]["songs"])
        
    def test_song_only_case_on_multiple_row(self):
        test_argument = pd.DataFrame(
                                   {'tags': [['락'], ["가"], ["나","다"]],
                                    'id': [61281,12345,12346],
                                    'plylst_title': ['여행같은 음악',"테스트1","테스트2"],
                                    'songs': [[525514, 129701, 383374, 562083, 297861,
                                               139541, 351214, 650298, 531057, 205238,
                                               706183, 127099, 660493, 461973, 121455,
                                               72552, 223955, 324992, 50104],
                                              [1,2,3,4,5],
                                              [6,7,8,9,10]],
                                    'like_cnt': [71,12,34],
                                    'updt_date': ['2013-12-19 18:36:19.000',
                                                  '2013-12-19 18:36:19.000',
                                                  '2013-12-19 18:36:19.000']}
                                   )
        expected = pd.DataFrame(
                                   {'tags': [[],[],[]],
                                    'id': [61281,12345,12346],
                                    'plylst_title': ['여행같은 음악',"테스트1","테스트2"],
                                    'songs': [[525514, 129701, 383374, 562083, 297861,
                                               139541, 351214, 650298, 531057],
                                              [1,2],
                                              [6,7]],
                                    'like_cnt': [71,12,34],
                                    'updt_date': ['2013-12-19 18:36:19.000',
                                                  '2013-12-19 18:36:19.000',
                                                  '2013-12-19 18:36:19.000']}
                                   )
        
        actual = DataGenerator()._mask(test_argument,["songs"], ["tags"])
        assert_series_equal(expected["tags"], actual[0]["tags"])
        assert len(expected["songs"]) == len(actual[0]["songs"])
    def test_song_only_sample_of_train_data(self):
        train = pd.read_json(stage1_config.TRAINING_DATA_FILE)
        test_argument = train.iloc[:20]
        actual = DataGenerator()._mask(test_argument,["songs"], ["tags"])
        
        
    def test_all_types_on_train_data(self):
        train = pd.read_json(stage1_config.TRAINING_DATA_FILE)
        test_argument = train
        DataGenerator()._mask(test_argument,["songs"], ["tags"])
        DataGenerator()._mask(test_argument,["tags"], ["songs"])
        DataGenerator()._mask(test_argument,[], ["songs","tags"])
        DataGenerator()._mask(test_argument,["songs","tags"], [])
    
    def test_song_only_start_from_index_not_0(self):
        train = pd.read_json(stage1_config.TRAINING_DATA_FILE)
        test_argument = train[1:20]
        DataGenerator()._mask(test_argument, ['songs', 'tags'], [])