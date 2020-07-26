# -*- coding: utf-8 -*-

import copy
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares as ALS

class Wrmf:
    def __init__(self, params):
        params = params.copy()
        self.c = params["c"]
        del params["c"]
        self.model = ALS(**params)
        self.plylst_feature_mapping = {}
        self.song_feature_mapping = {}
        self.n_plylst = None
        self.n_song = None
        self.n_tag = None
        self.n_train = None
        self.n_test = None
        
        
    def _make_features_mapping(self, df, song_meta):
        plylst_feature_list = []
        plylst_feature_list += list(set(df["songs"].explode().dropna()))
        plylst_feature_list += list(set(df["tags"].explode().dropna()))
        for plylst_feature in plylst_feature_list:
            self.plylst_feature_mapping.setdefault(plylst_feature,len(self.plylst_feature_mapping))
        
        song_feature_list = []
        song_feature_list += list(df["id"])
        for song_feature in song_feature_list:
            self.song_feature_mapping.setdefault(song_feature,len(self.song_feature_mapping))
    
    def _make_user_item_matrix(self,df, song_meta):
        mat_for_csr = pd.DataFrame(columns = ["row", "col"])
        
        temp = df[["id", "songs"]].explode("songs")
        temp.columns = ["row", "col"]
        mat_for_csr = pd.concat([mat_for_csr, temp])
        
        temp = df[["id", "tags"]].explode("tags")
        temp.columns = ["row", "col"]
        mat_for_csr = pd.concat([mat_for_csr, temp])
        
        mat_for_csr = mat_for_csr.dropna().reset_index(drop = True)
        mat_for_csr["row"] = mat_for_csr["row"].map(self.song_feature_mapping, na_action = "ignore")
        mat_for_csr["col"] = mat_for_csr["col"].map(self.plylst_feature_mapping, na_action = "ignore")
        
        return mat_for_csr
    
    def preprocess(self, train, test, song_meta):
        total = pd.concat([train, test]).reset_index(drop = True)
        self.n_train = len(train)
        self.n_test = len(test)
        self.n_song = len(set(total["songs"].explode().dropna()))
        self.n_tag = len(set(total["tags"].explode().dropna()))
        self.n_plylst = len(total["id"])
    
        song_meta = song_meta.loc[song_meta["id"].isin(set(total["songs"].explode()))]
        self._make_features_mapping(total, song_meta)
        
        user_item_matrix = self._make_user_item_matrix(total, song_meta)
        user_item_matrix_csr = csr_matrix((np.ones(user_item_matrix.shape[0]),
                                           (user_item_matrix["row"],user_item_matrix["col"])),
                                          shape = [len(self.song_feature_mapping),
                                                   len(self.plylst_feature_mapping)])                                  
        return user_item_matrix_csr
    
    def fit(self, processed_total):
        f = "wrmf_model.npz"
        if f not in os.listdir("model"):
            self.model.fit(self.c * processed_total.T)
            np.savez("model/" + f, user_factors = self.model.user_factors,
                        item_factors = self.model.item_factors)
        else:
            weights = np.load("model/" + f)
            self.model.user_factors = weights["user_factors"]
            self.model.item_factors = weights["item_factors"]
            weights.close()
    
    def recommend(self, processed_total, num_songs, num_tags):
        song_model = copy.copy(self.model)
        tag_model = copy.copy(self.model)

        song_model.user_factors = self.model.user_factors[-self.n_test:]
        song_model.item_factors = self.model.item_factors[:self.n_song]
        
        tag_model.user_factors = self.model.user_factors[-self.n_test:]
        tag_model.item_factors = self.model.item_factors[-self.n_tag:]
        
        song_rec_csr = processed_total[-self.n_test:,
                                       :self.n_song]
        tag_rec_csr = processed_total[-self.n_test:,
                                       -self.n_tag:]
    
        plylst_mapping = {i:v for i, v in zip(np.arange(self.n_plylst), list(self.song_feature_mapping.keys())[:self.n_plylst])}
        song_mapping = {i:v for i, v in zip(np.arange(self.n_song), list(self.plylst_feature_mapping.keys())[:self.n_song])}
        tag_mapping = {i:v for i, v in zip(np.arange(self.n_tag), list(self.plylst_feature_mapping.keys())[-self.n_tag:])}
        
        song_rec_df = pd.DataFrame(columns=["plylst_id", "song_id", "score"])
        tag_rec_df = pd.DataFrame(columns=["plylst_id", "tag", "score"])
        for u in range(self.n_test):
            song_rec = song_model.recommend(u, song_rec_csr, N = num_songs)
            song_ids = [id_ for id_, _ in song_rec]
            song_scores = [score for _, score in song_rec]
            song_plylst_ids = np.repeat(self.n_train + u, num_songs)
            song_recommended = pd.DataFrame({"plylst_id" : song_plylst_ids,
                                             "song_id" : song_ids,
                                             "score" : song_scores})
            song_rec_df = pd.concat([song_rec_df, song_recommended])
            
            tag_rec = tag_model.recommend(u, tag_rec_csr, N = num_tags)
            tag_ids = [id_ for id_, _ in tag_rec]
            tag_scores = [score for _, score in tag_rec]
            tag_plylst_ids = np.repeat(self.n_train + u, num_tags)
            tag_recommended = pd.DataFrame({"plylst_id" : tag_plylst_ids,
                                             "tag" : tag_ids,
                                             "score" : tag_scores})
            tag_rec_df = pd.concat([tag_rec_df, tag_recommended])
            
        song_rec_df["plylst_id"] = song_rec_df["plylst_id"].map(plylst_mapping)
        song_rec_df["song_id"] = song_rec_df["song_id"].map(song_mapping)
        song_rec_df = song_rec_df.reset_index(drop=True)
        
        tag_rec_df["plylst_id"] = tag_rec_df["plylst_id"].map(plylst_mapping)
        tag_rec_df["tag"] = tag_rec_df["tag"].map(tag_mapping)
        tag_rec_df = tag_rec_df.reset_index(drop=True)
        
        return song_rec_df, tag_rec_df

if __name__ == "__main__":
    pass