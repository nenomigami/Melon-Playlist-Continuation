# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:33:28 2020

@author: ghdbs
"""
import pytest
import dill as pkl
import pandas as pd
import numpy as np
import stage1_config
from src.modeling.wrmf import Wrmf
from numpy.testing import *
class TestWrmf(object):
    
    def test_shape_after_fit_on_whole_traning(self):
        with open(stage1_config.STAGE1_PROCESSED_TRAIN_FILE,"rb") as f:
            train = pkl.load(f)
        
        params = {"factors": 320, "regularization": 0.0001, "c": 150, "use_gpu" : True}
        nunique_feature = {"songs" : 626251 , "tags" : 28964}
        test_argument = train
        wrmf = Wrmf(params, nunique_feature)
        actual = wrmf.fit(train)
        

    def test_song_model_is_the_same_with_part_of_model(self):
        
        with open(stage1_config.SAVE_FOLDER + "model1.pkl","rb") as f:
            model = pkl.load(f)
        
        actual_song_user_factor = model.song_model.user_factors
        actual_song_item_factor = model.song_model.item_factors
        actual_tag_user_factor = model.tag_model.user_factors
        actual_tag_item_factor = model.tag_model.item_factors
        
        expected_org_user_factor = model.model.user_factors
        expected_org_item_factor = model.model.item_factors
        
        assert_array_equal(actual_song_user_factor, expected_org_user_factor)
        assert_array_equal(actual_song_item_factor, expected_org_item_factor[:626251])
        assert_array_equal(actual_tag_user_factor, expected_org_user_factor)
        assert_array_equal(actual_tag_item_factor, expected_org_item_factor[-28964:])

            
    def test_fit_result_having_same_data_with_original(self):
        with open(stage1_config.STAGE1_PROCESSED_TRAIN_FILE,"rb") as f:
            train = pkl.load(f)
        
        params = {"factors": 320, "regularization": 0.0001, "c": 150, "use_gpu" : True}
        nunique_feature = {"songs" : 626251 , "tags" : 28964}
        wrmf = Wrmf(params, nunique_feature)
        model = wrmf.fit(train)
        test = pd.concat(
                        [pd.read_json(f) for f in stage1_config.STAGE1_VALID_Q]
                        ).reset_index(drop = True)
        idx = np.arange(train.shape[0]-len(test),train.shape[0])[0:10]
        ten_song_len_in_df = test.iloc[0:10].apply(lambda r : len(r["songs"]),axis = 1).to_numpy()
        ten_song_len_in_csr = model.song_rec_csr[idx,:].toarray().sum(axis=1)
        ten_tag_len_in_df = test.iloc[0:10].apply(lambda r : len(r["tags"]),axis = 1).to_numpy()
        ten_tag_len_in_csr = model.tag_rec_csr[idx,:].toarray().sum(axis=1)
        
        assert_array_equal(ten_song_len_in_df, ten_song_len_in_csr)
        assert_array_equal(ten_tag_len_in_df, ten_tag_len_in_csr)

    def test_predict_on_sample_valid(self):
        with open(stage1_config.STAGE1_PROCESSED_TRAIN_FILE,"rb") as f:
            train = pkl.load(f)
        test = pd.concat(
                        [pd.read_json(f) for f in stage1_config.STAGE1_VALID_Q]
                        ).reset_index(drop = True)
        with open(stage1_config.SAVE_FOLDER + "model1.pkl","rb") as f:
            model = pkl.load(f)

        assert len(model.song_model.recommend(133978, model.song_rec_csr, N = 1000000)) \
            == 626251 - len(test.iloc[0].songs)
        assert len(model.song_model.recommend(133979, model.song_rec_csr, N = 1000000)) \
            == 626251 - len(test.iloc[1].songs)
        assert len(model.tag_model.recommend(133978, model.tag_rec_csr, N = 100000)) \
            == 28964 - len(test.iloc[0].tags)
        assert len(model.tag_model.recommend(133979, model.tag_rec_csr, N = 100000)) \
            == 28964 - len(test.iloc[1].tags)
