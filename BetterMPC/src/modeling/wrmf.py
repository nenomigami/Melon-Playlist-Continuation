# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:32:46 2020

@author: ghdbs
"""
import pandas as pd
import numpy as np
import dill as pkl
import stage1_config
from implicit.als import AlternatingLeastSquares as ALS

class Wrmf:
    
    def __init__(self, params = {"c" : None}, nunique_feature = None):
        self.params = params.copy()
        self.c = params["c"]
        del params["c"]
        self.model = ALS(**params)
        self.song_model = ALS(**params)
        self.tag_model = ALS(**params)
        self.song_rec_csr = None
        self.tag_rec_csr = None
        self.nunique_feature = nunique_feature
        
        
    def fit(self, X):
        self.model.fit(self.c * X.T)
        self.song_model.user_factors = self.model.user_factors
        self.song_model.item_factors = self.model.item_factors[:self.nunique_feature["songs"]]
        
        self.tag_model.user_factors = self.model.user_factors
        self.tag_model.item_factors = self.model.item_factors[-self.nunique_feature["tags"]:]
        
        self.song_rec_csr = X[:,:self.nunique_feature["songs"]]
        self.tag_rec_csr = X[:,-self.nunique_feature["tags"]:]
        #save_model 로 따로 두는 것이 좋음
            
        return self
    
    def predict(self, idx, num_songs, num_tags):
        song_rec_df = pd.DataFrame()
        tag_rec_df = pd.DataFrame()
        for u in idx:
            song_rec = self.song_model.recommend(u, self.song_rec_csr, N = num_songs)
            song_ids = [id_ for id_, _ in song_rec]
            song_scores = [score for _, score in song_rec]
            song_plylst_ids = np.repeat(u, num_songs)
            song_recommended = pd.DataFrame({"plylst_id" : song_plylst_ids,
                                             "song_id" : song_ids,
                                             "score" : song_scores})
            song_rec_df = pd.concat([song_rec_df, song_recommended])
            
            tag_rec = self.tag_model.recommend(u, self.tag_rec_csr, N = num_tags)
            tag_ids = [id_ for id_, _ in tag_rec]
            tag_scores = [score for _, score in tag_rec]
            tag_plylst_ids = np.repeat(u, num_tags)
            tag_recommended = pd.DataFrame({"plylst_id" : tag_plylst_ids,
                                             "tag" : tag_ids,
                                             "score" : tag_scores})
            tag_rec_df = pd.concat([tag_rec_df, tag_recommended])        
            
        return song_rec_df, tag_rec_df
        
    def save_model(self, save_file):
        with open(stage1_config.SAVE_FOLDER + save_file + ".pkl", "wb") as f:
            pkl.dump(self, f)
        with open(stage1_config.SAVE_FOLDER + save_file + "_config.txt", "a") as f:
            f.write(str(self.params))
                
    def load_model(self, save_file):
        with open(stage1_config.SAVE_FOLDER + save_file + ".pkl","rb") as f:
            self = pkl.load(f)
        return self
                
if __name__ == "__main__":
    pass 