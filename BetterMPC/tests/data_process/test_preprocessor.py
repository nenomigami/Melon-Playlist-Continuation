# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 01:12:46 2020

@author: ghdbs
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import stage1_config
from src.data_process.preprocessor import Uir_matrix_maker, Csr_matrix_maker, Stage2_dataset_builder
from pandas.testing import *
from numpy.testing import *
import dill as pkl
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
        expected = 5505563 + 493184
        
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
        actual = Csr_matrix_maker().fit_transform(test_argument).shape[1]
        expected = sum(uir_matrix_maker.nunique_features.values())
        assert actual == expected
        
    def test_length_on_train(self):
        uir_matrix_maker = Uir_matrix_maker(stage1_config.TARGET_VARS)
        test_argument = pd.read_json(stage1_config.TRAINING_DATA_FILE)
        test_argument = uir_matrix_maker.fit_transform(test_argument)
        actual = Csr_matrix_maker().fit_transform(test_argument).shape[1]
        expected = sum(uir_matrix_maker.nunique_features.values())
        assert test_argument.isna().sum().sum() == 0 
        assert actual == expected
    
    def test_first_row_is_the_same_with_that_of_uir(self):
        uir_matrix_maker = Uir_matrix_maker(stage1_config.TARGET_VARS)
        test_argument = pd.concat(
                        [pd.read_json(f) for f in stage1_config.STAGE1_TRAIN]
                        ).reset_index(drop = True)
        test_argument = uir_matrix_maker.fit_transform(test_argument)
        actual = sum(Csr_matrix_maker().fit_transform(test_argument)[0,].data)
        expected = len(test_argument.groupby("id", sort=False).\
                       apply(lambda grp : [x for x in grp["songs_tags"]]).iloc[0])
        assert actual == expected

class TestStage2DatasetBuilder(object):
    
    def test_dataset_builder_having_right_target_on_one_sample(self):
        with open(stage1_config.PROCESSED_DATA_FOLDER + "temp.pkl", "rb") as f:
            song_rec_df, tag_rec_df = pkl.load(f)
        stage2_dataset_builder = Stage2_dataset_builder()
        song_test_argument = song_rec_df.iloc[:50]
        tag_test_argument = tag_rec_df.iloc[:50]
        ans = pd.concat(
                    [pd.read_json(f) for f in stage1_config.STAGE1_VALID_A]
                    ).reset_index(drop = True).head(1)
        song_test_argument, tag_test_argument = stage2_dataset_builder.transform(song_test_argument,
                                                                                 tag_test_argument,
                                                                                 ans)
        song_actual = song_test_argument["target"].sum()
        song_expected = 3
        tag_actual = tag_test_argument["target"].sum()
        tag_expected = 2
        
        assert song_actual == song_expected
        assert tag_actual == tag_expected
        
    def test_dataset_builder_having_right_target_on_samples(self):
        with open(stage1_config.PROCESSED_DATA_FOLDER + "temp.pkl", "rb") as f:
            song_rec_df, tag_rec_df = pkl.load(f)
        stage2_dataset_builder = Stage2_dataset_builder()
        song_test_argument = song_rec_df.groupby("plylst_id", sort=False).head(50).iloc[:1000]
        tag_test_argument = tag_rec_df.groupby("plylst_id", sort=False).head(50).iloc[:1000]
        ans = pd.concat(
                        [pd.read_json(f) for f in stage1_config.STAGE1_VALID_A]
                        ).reset_index(drop = True).iloc[:20]
        song_test_argument ,tag_test_argument = stage2_dataset_builder.transform(song_test_argument,
                                                                                 tag_test_argument
                                                                                 , ans)
        song_actual = song_test_argument[["plylst_id","target"]].groupby("plylst_id",
                                                                         sort = False).sum().target.tolist()
        """
        target1_by_plylst = []
        for grp in tag_test_argument[["plylst_id","tag"]].groupby("plylst_id",
                                                                       sort = False):
            print(grp[0])
            target = 0 
            for song in grp[1]["tag"]:
                if song in ans.loc[ans["id"] == grp[0]].tags.iloc[0]:
                    target += 1
            target1_by_plylst.append(target)
        """
        song_expected = [3, 2, 4, 0, 0, 1, 3, 6, 11, 6, 0, 13, 1, 0, 0, 2, 5, 19, 3, 5]
        tag_actual = tag_test_argument[["plylst_id","target"]].groupby("plylst_id",
                                                                         sort = False).sum().target.tolist()
        
        tag_expected = [2, 1, 0, 1, 0, 2, 2, 5, 4, 1, 1, 2, 1, 1, 0, 2, 6, 4, 1, 2]
        
        assert song_actual == song_expected
        assert tag_actual == tag_expected