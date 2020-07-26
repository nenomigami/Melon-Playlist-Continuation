# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:48:38 2020

@author: ghdbs
"""
import copy
import os
from pathlib import Path
import numpy as np
import pandas as pd
import dill as pkl
from catboost import CatBoostClassifier, Pool

#with open("song_candidates.pkl","rb") as f:
#    song_rec_df, tag_rec_df = pkl.load(f)

class lightgbm:
    def __init__(self):
        pass
    def preprocess(self):
        pass
    
if __name__ == "__main__":
    pass
DATA_FOLDER = Path("data/fold0")

FTRAIN = DATA_FOLDER  / Path("train.json")
FVALID1 = DATA_FOLDER / Path("val1_q.json")
FVALID2 = DATA_FOLDER / Path("val2_q.json")
FTEST = Path("res/test.json")

FVALID1_A = DATA_FOLDER / Path("val1_a.json")
FVALID2_A = DATA_FOLDER / Path("val2_a.json")
SONG_META = Path("res/song_meta.json")
song_meta = pd.read_json(SONG_META)

stage1_train = pd.concat([pd.read_json(FTRAIN),pd.read_json(FTEST)]).reset_index(drop=True)
stage2_train_idx = pd.read_json(FVALID1)["id"]
stage2_valid_idx = pd.read_json(FVALID2)["id"]


PLAYLIST_ID = "plylst_id"
SONG_ID = "song_id"

def merge_meta(uif, org, song_meta):
    def fastleftmerge(left, right, left_on, right_on):
        return left.merge(right.rename({right_on : left_on}, axis = 1))
    merged = fastleftmerge(uif, org, left_on = PLAYLIST_ID, right_on = "id")
    merged = fastleftmerge(merged, song_meta, left_on = SONG_ID, right_on = "id")
    return merged

def make_target(uif):
    df = uif.copy()
    df["target"] = df.apply(lambda r: 1 if r[SONG_ID] in r["songs"] else 0, axis = 1)
    return df

def drop_songs_after_plyst(df, song_updt_date, plyst_updt_date):
    def clean_zero_date(date):
      date = list(str(date))
      if date[4:6] == ["0","0"]:
        date[4:6] = ["0","1"]
      if date[6:8] == ["0","0"]:
        date[6:8] = ["0","1"]
      if date == ["0"]:
        date = ["1","9","0","0","0","1","0","1"]
      return "".join(date)
    df = df.copy()
    df[song_updt_date] = df[song_updt_date].map(lambda date : clean_zero_date(date))
    df[song_updt_date] = pd.to_datetime(df[song_updt_date], format = "%Y%m%d")
    df[plyst_updt_date] = pd.to_datetime(df[plyst_updt_date])
    df = df.loc[df[song_updt_date] <= df[plyst_updt_date]]
    return df
#latent feature / 1등 답 참조
total_valid = pd.concat([pd.read_json(FVALID1_A), pd.read_json(FVALID2_A)]).reset_index(drop = True)
total = merge_meta(song_rec_df, total_valid, song_meta)
del song_rec_df, tag_rec_df
total = make_target(total)
"""import dill as pkl
with open("working.pkl","wb") as f:
    pkl.dump(total, f)
import dill as pkl
with open("working.pkl","rb") as f:
    total = pkl.load(f)
"""
#song_frequency
song_freq = stage1_train["songs"].explode().value_counts().\
            reset_index().rename(columns={"index": "song_id", "songs": "song_frequency"})
total = total.merge(song_freq, on ="song_id")

#artist_frequency
artist_freq = total[["song_id","artist_name_basket"]].explode("artist_name_basket")\
    .drop_duplicates().reset_index(drop=True)
temp = total["artist_name_basket"].explode().value_counts().reset_index()\
        .rename(columns={"index": "artist_name_basket", "artist_name_basket": "artist_frequency"})
temp.loc[temp["artist_name_basket"] == "Various Artists", "artist_frequency"] = 1
artist_freq = artist_freq.merge(temp, on = "artist_name_basket").drop("artist_name_basket", axis = 1)
artist_freq = artist_freq.groupby("song_id").mean().reset_index()

total = total.merge(artist_freq, on ="song_id")

#various artsts
total["Various_Artists"] = total["artist_name_basket"].map(lambda r : 1 if "Various Artists" in r else 0).sum()


total.columns
total = drop_songs_after_plyst(total, "issue_date", "updt_date")
total = total.drop(["plylst_title","album_name", "album_id", "artist_id_basket",
                    "song_name", "song_gn_gnr_basket", "artist_name_basket"],axis = 1)

#

#a해당하는 plylsy_id 에 예측한 song_id가 있으면 1 없으면 0
stage2_train = 
stage2_valid = pd.concat([pd.read_json(FVALID1),pd.read_json(FVALID2)]).reset_index(drop=True)
stage2_valid_a = pd.concat([pd.read_json(FVALID1_A), pd.read_json(FVALID2_A)])[["id","tags","songs"]]
song_meta = pd.read_json(SONG_META)
