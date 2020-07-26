# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:00:22 2020

@author: ghdbs
"""

from algorithms.Wrmf import Wrmf 
from pathlib import Path
import pandas as pd
import numpy as np
from util import calculate_song_recall
import copy
import os
import dill as pkl
from catboost import CatBoostClassifier, Pool
import re

SEED = 1000

#DATA_FOLDER = Path("data/fold0")

#FTRAIN = DATA_FOLDER  / Path("train.json")
#FVALID1 = DATA_FOLDER / Path("val1_q.json")
#FVALID2 = DATA_FOLDER / Path("val2_q.json")
#FTEST = Path("res/test.json")

#FVALID1_A = DATA_FOLDER / Path("val1_a.json")
#FVALID2_A = DATA_FOLDER / Path("val2_a.json")
SONG_META = Path("res/song_meta.json")

PLAYLIST_ID = "plylst_id"
SONG_ID = "song_id"

#stage1_train = pd.concat([pd.read_json(FTRAIN),pd.read_json(FTEST)]).reset_index(drop=True)
#stage1_valid = pd.concat([pd.read_json(FVALID1),pd.read_json(FVALID2)]).reset_index(drop=True)
#stage1_valid_a = pd.concat([pd.read_json(FVALID1_A), pd.read_json(FVALID2_A)])[["id","tags","songs"]].reset_index(drop=True)

DATA_FOLDER = Path("res")
FTRAIN = DATA_FOLDER  / Path("train.json")
FVALID = DATA_FOLDER / Path("val.json")
FTEST = Path("res/test.json")

stage1_train = pd.concat([pd.read_json(FTRAIN),
                          pd.read_json(FTEST)]).reset_index(drop=True)
#stage1_train = pd.concat([pd.read_json(FTRAIN),
#                          pd.read_json(FVALID1),
#                          pd.read_json(FVALID2)]).reset_index(drop=True)
stage1_valid = pd.read_json(FVALID)
#stage2_train_idx = pd.read_json(FVALID1)["id"]
#stage2_valid_idx = pd.read_json(FVALID2)["id"]
song_meta = pd.read_json(SONG_META)

params = {"factors": 320, "regularization": 0.0001, "c": 150}
model = Wrmf(params)
processed_total = model.preprocess(stage1_train, stage1_valid, song_meta)
model.fit(processed_total)
#song_rec_df, tag_rec_df = model.recommend(processed_total, 1000, 10)
#with open("final.pkl","rb") as f:
#    (song_rec_df, tag_rec_df) = pkl.load(f)
#with open("song_candidates.pkl","rb") as f:
#    song_rec_df, tag_rec_df = pkl.load(f)
with open("final_valid.pkl","rb") as f:
    song_rec_df, tag_rec_df = pkl.load(f)

def merge_meta(uif, org, song_meta):
    def fastleftmerge(left, right, left_on, right_on):
        return left.merge(right.rename({right_on : left_on}, axis = 1))
    merged = fastleftmerge(uif, org, left_on = PLAYLIST_ID, right_on = "id")
    merged = fastleftmerge(merged, song_meta, left_on = SONG_ID, right_on = "id")
    return merged

def make_target(uif, type = "song"):
    df = uif.copy()
    total_valid = pd.concat([pd.read_json(FVALID1_A),\
                             pd.read_json(FVALID2_A)]).reset_index(drop = True)\
                            [["id", "songs", "tags"]]
    df = df.merge(total_valid, left_on = PLAYLIST_ID, right_on = "id")
    if type == "song":
        uif["target"] = df.apply(lambda r: 1 if r[SONG_ID] in r["songs"] else 0, axis = 1)
    if type == "tag":
        uif["target"] = df.apply(lambda r: 1 if r["tag"] in r["tags"] else 0, axis = 1)
    return uif

def add_song_freq(df, stage1_train):
    song_freq = stage1_train["songs"].explode().value_counts().\
                reset_index().rename(columns={"index": "song_id", "songs": "song_frequency"})
    df = df.merge(song_freq, on ="song_id", how = "left").fillna(1)
    return df

def add_artist_freq(df):
    artist_freq = df[["song_id","artist_name_basket"]].explode("artist_name_basket")\
                    .drop_duplicates().reset_index(drop=True)
    temp = df["artist_name_basket"].explode().value_counts().reset_index()\
                    .rename(columns={"index": "artist_name_basket", "artist_name_basket": "artist_frequency"})
    temp.loc[temp["artist_name_basket"] == "Various Artists", "artist_frequency"] = 1
    artist_freq = artist_freq.merge(temp, on = "artist_name_basket").drop("artist_name_basket", axis = 1)
    artist_freq = artist_freq.groupby("song_id").mean().reset_index()
    df = df.merge(artist_freq, on ="song_id")
    return df
def add_song_gr(df):
    temp = df["song_gn_gnr_basket"].explode().reset_index().\
                rename(columns={"index": "index", "song_gn_gnr_basket": "gr"}).\
                        pivot_table(index=['index'], columns=["gr"], aggfunc=[len], fill_value=0)
    temp.columns = temp.columns.droplevel()
    temp = temp.astype("int8")
    return temp

#해당하는 plylsy_id 에 예측한 song_id가 있으면 1 없으면 0
#total = make_target(song_rec_df)
#total_train = pd.concat([pd.read_json(FVALID1), pd.read_json(FVALID2)]).reset_index(drop = True)
total = song_rec_df; del song_rec_df
total_train = stage1_valid
total = merge_meta(total, total_train, song_meta)
#del song_rec_df#,tag_rec_df

#song_frequency
total = add_song_freq(total, stage1_train)

#artist_frequency
total = add_artist_freq(total)

#various artsts
total["Various_Artists"] = total["artist_name_basket"].map(lambda r : 1 if "Various Artists" in r else 0)

#drop_song_after_plylst
total["updt_date"] = total["updt_date"].map(lambda r : int(re.sub("[^0-9]", "",r[0:10])))
#total = (total.loc[total["issue_date"] <= total["updt_date"]]).reset_index(drop=True)
#plylst_len
total["plylst_len"] = total["plylst_title"].map(len)

total["song_len"] = total["song_name"].map(len)
#ntags
total["n_tags"] = total["tags"].map(len)

drop_list = ["tags", "plylst_title", "songs","song_gn_dtl_gnr_basket", "album_name",
             "song_name", "album_id", "artist_id_basket", "artist_name_basket"]
total = total.drop(drop_list,axis = 1)
"""
import dill as pkl
with open("working.pkl","wb") as f:
    pkl.dump(total, f)

import dill as pkl
with open("working.pkl","rb") as f:
    total = pkl.load(f)
"""
#generanl gerne
total = pd.concat([add_song_gr(total),total], axis = 1).fillna(0)
total = total.drop("song_gn_gnr_basket", axis = 1)

total = total.sort_values(by = "plylst_id").reset_index(drop = True)
total
import gc; gc.collect()
"""
import dill as pkl
with open("working1.pkl","wb") as f:
    pkl.dump(total.iloc[:13015000], f)
with open("working2.pkl","wb") as f:
    pkl.dump(total.iloc[13015000:], f)
"""

#import dill as pkl
#with open("working1.pkl","rb") as f:
#    total = pkl.load(f)
with open("working2.pkl","rb") as f:
    total = pkl.load(f)
"""
with open("working1.pkl","wb") as f:
    pkl.dump(total.iloc[:6515000], f)

with open("working1.5.pkl","wb") as f:
    pkl.dump(total.iloc[6515000:], f)
"""
#stage2_train = total.loc[total["plylst_id"].isin(np.arange(100))]
#stage2_valid = total.loc[total["plylst_id"].isin(np.arange(100,200))]
#stage2_train = total.loc[total["plylst_id"].isin(stage2_train_idx)].reset_index(drop=True)
#stage2_valid = total.loc[total["plylst_id"].isin(stage2_valid_idx)].reset_index(drop=True)
features = [c for c in total.columns if c not in ['target']]
#del total, song_meta

#trn_data = Pool(data = stage2_train.drop("target", axis = 1).values,
#                label = stage2_train["target"])
#val_data = Pool(data = stage2_valid.drop("target", axis = 1).values,
#                label = stage2_valid["target"])

model = CatBoostClassifier(iterations = 8900,
                           task_type = "GPU",
                           devices='0',
                           random_seed = SEED)
model.load_model("model/catboost.cbm")

#fit_model = model.fit(trn_data,
#                    eval_set = val_data,
#                    use_best_model = True,
#                     verbose = 1000,
#                     early_stopping_rounds = 300,
#                     plot = False)

def to_result(fit_model, test):
    result = pd.DataFrame()
    p_valid = pd.DataFrame()
    p_valid["prob"] = fit_model.predict_proba(test[features].values)[:,1]
    p_valid["plylst_id"] = test["plylst_id"].reset_index(drop=True)
    p_valid["song_id"] = test["song_id"].reset_index(drop=True)
    top100 = p_valid.sort_values(by = "prob", ascending = False).\
        groupby("plylst_id", sort = False).head(100).reset_index(drop=True)
    result["id"] = top100["plylst_id"].drop_duplicates().reset_index(drop=True)
    result["songs"] = top100.groupby("plylst_id", sort = False).apply(lambda grp:[id for id in grp["song_id"]]).reset_index(drop = True)
    return result
"""
with open("song_result.pkl", "wb") as f:
    pkl.dump(result, f)

with open("song_result.pkl", "rb") as f:
    result1 = pkl.load(f)
result = to_result(model, total)
    
"""
result = pd.DataFrame()
p_valid = pd.DataFrame()
p_valid["prob"] = model.predict_proba(total[features].values)[:,1]
p_valid["plylst_id"] = total["plylst_id"].reset_index(drop=True)
p_valid["song_id"] = total["song_id"].reset_index(drop=True)
top100 = p_valid.sort_values(by = "prob", ascending = False).\
    groupby("plylst_id", sort = False).head(100).reset_index(drop=True)
result["id"] = top100["plylst_id"].drop_duplicates().reset_index(drop=True)
result["songs"] = top100.groupby("plylst_id", sort = False).apply(lambda grp:[id for id in grp["song_id"]]).reset_index(drop = True)
result = pd.concat([result1, result]).reset_index(drop=True)
#result = to_result(model, stage2_valid)
result = result.sort_values("id").reset_index(drop = True)

result["tags"] = tag_rec_df.groupby("plylst_id").\
                    apply(lambda grp:[id for id in grp["tag"]]).\
                        reset_index(drop = True)
result.to_json("results.json", orient="records")
temp = result.copy()
temp.index = temp.id
temp.columns = ['id', 'songs', 'tags']
temp.to_json("results.json", orient="records")
"""
import dill as pkl
with open("song_result.pkl","wb") as f:
    pkl.dump(result, f)

import dill as pkl
with open("song_candidates.pkl","wb") as f:
    pkl.dump((song_rec_df, tag_rec_df), f)

def to_result(song_uir_matrix, tag_uir_matrix):
    import json
    result = pd.DataFrame()
    result["id"] = song_uir_matrix["plylst_id"].drop_duplicates().reset_index(drop = True)
    result["songs"] = song_uir_matrix.groupby("plylst_id", sort = False).apply(lambda grp: 
                                                                 [id for id in grp["song_id"]]).reset_index(drop = True)
    result["tags"] = tag_uir_matrix.groupby("plylst_id", sort = False).apply(lambda grp: 
                                                                 [id for id in grp["tag"]]).reset_index(drop = True)    
    return result

result = to_result(song_rec_df, tag_rec_df)
result.songs
print(calculate_song_recall(stage1_valid, result, stage1_valid_a))

total = make_target(tag_rec_df, "tag")
plylst_meta = total_train[["id","songs"]].explode("songs").\
    rename(columns={"id" : "plylst_id", "songs": "song_id"}).\
        merge(song_meta, left_on = "song_id", right_on = "id",
              how = "left")

def add_artist_freq_tag(df, plylst_meta):
    artist_freq = plylst_meta[["song_id","artist_name_basket"]].explode("artist_name_basket")\
                    .drop_duplicates().reset_index(drop=True)
    temp = plylst_meta["artist_name_basket"].explode().value_counts().reset_index()\
                    .rename(columns={"index": "artist_name_basket", "artist_name_basket": "artist_frequency"})
    temp.loc[temp["artist_name_basket"] == "Various Artists", "artist_frequency"] = 1
    artist_freq = artist_freq.merge(temp, on = "artist_name_basket").drop("artist_name_basket", axis = 1)
    artist_freq = artist_freq.groupby("song_id").mean().reset_index()
    plylst_meta = plylst_meta.merge(artist_freq, on ="song_id")
    artist_freq = plylst_meta[["plylst_id", "artist_frequency"]].groupby("plylst_id", sort = False).mean().reset_index()
    df = df.merge(artist_freq, on ="plylst_id", how = "outer").fillna(-999)
    return df
total = add_artist_freq_tag(total, plylst_meta)

def add_song_freq_tag(df, plylst_meta):
    song_freq = total_train["songs"].explode().value_counts().\
                    reset_index().rename(columns={"index": "song_id", "songs": "song_frequency"})
    plylst_meta = plylst_meta.merge(song_freq, on ="song_id")
    song_freq = plylst_meta[["plylst_id", "song_frequency"]].groupby("plylst_id", sort = False).mean().reset_index()
    df = df.merge(song_freq, on ="plylst_id", how = "outer").fillna(-999)
    return df

total = add_song_freq_tag(total, plylst_meta)

total["plylst_len"] = total.merge(total_train[["id","plylst_title"]],\
                                  left_on = "plylst_id", right_on = "id")["plylst_title"].map(len)
#ntags
total["n_tags"] = total["tags"].map(len)

def add_gnr_gr(df, plylst_meta):
    temp = pd.crosstab(plylst_meta[["song_gn_gnr_basket","plylst_id"]].explode("song_gn_gnr_basket")["plylst_id"],
                       plylst_meta[["song_gn_gnr_basket","plylst_id"]].explode("song_gn_gnr_basket")["song_gn_gnr_basket"],
                       normalize = "index", dropna = False).reset_index()
    df = df.merge(temp, on = "plylst_id", how = "left").fillna(-999)
    return df

total = add_gnr_gr(total, plylst_meta)

stage2_train = total.loc[total["plylst_id"].isin(stage2_train_idx)].reset_index(drop=True)
stage2_valid = total.loc[total["plylst_id"].isin(stage2_valid_idx)].reset_index(drop=True)
features = [c for c in total.columns if c not in ['target']]

trn_data = Pool(data = stage2_train.drop("target", axis = 1),
                label = stage2_train["target"],
                cat_features = ["tag"])
val_data = Pool(data = stage2_valid.drop("target", axis = 1),
                label = stage2_valid["target"],
                cat_features = ["tag"])

model = CatBoostClassifier(iterations = 10000,
                           task_type = "GPU",
                           devices='0',
                           random_seed = SEED)

fit_model = model.fit(trn_data,
                     eval_set = val_data,
                     use_best_model = True,
                     verbose = 1000,
                     early_stopping_rounds = 300,
                     plot = False)

def to_result(fit_model, test):
    result = pd.DataFrame()
    p_valid = pd.DataFrame()
    p_valid["prob"] = fit_model.predict_proba(test[features].values)[:,1]
    p_valid["plylst_id"] = test["plylst_id"].reset_index(drop=True)
    p_valid["tag"] = test["tag"].reset_index(drop=True)
    top100 = p_valid.sort_values(by = "prob", ascending = False).\
        groupby("plylst_id", sort = False).head(10).reset_index(drop=True)
    result["id"] = top100["plylst_id"].drop_duplicates().reset_index(drop=True)
    result["tag"] = top100.groupby("plylst_id", sort = False).apply(lambda grp:[id for id in grp["tag"]]).reset_index(drop = True)
    return result

result = to_result(fit_model, stage2_valid)
result = result.sort_values("id")
import dill as pkl
with open("song_result.pkl","rb") as f:
    song_result = pkl.load(f)
song_result["tags"] = result["tag"]
song_result = song_result.reset_index(drop = True)
song_result.to_json("ret1.json", orient="records")

from evaluate2 import Evaluator
eva = Evaluator()
eva.evaluate("data/fold0/val2_a.json","ret1.json")
"""
0.341709 * 0.15 + 0.85 * 0.224708
