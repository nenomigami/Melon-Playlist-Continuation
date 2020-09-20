# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:40:34 2020

@author: ghdbs
"""

import pytest
import dill as pkl
import pandas as pd
import numpy as np
import stage1_config
from src.modeling.wrmf import Wrmf

def recommend(pred, pipeline):
    song_rec_df, tag_rec_df = pred[0].copy(), pred[1].copy()
    nunique_features = pipeline["uir_matrix_maker"].nunique_features
    song_tag_cat = pipeline["csr_matrix_maker"].songs_tags_c.categories
    id_cat = pipeline["csr_matrix_maker"].id_c.categories
    ix_to_song_id = {k:v for k,v in zip(\
                                        np.arange(len(song_tag_cat[:nunique_features["songs"]])),\
                                       song_tag_cat[:nunique_features["songs"]])}
    ix_to_tag_id = {k:v for k,v in zip(\
                                        np.arange(len(song_tag_cat[-nunique_features["tags"]:])),\
                                       song_tag_cat[-nunique_features["tags"]:])}
    ix_to_id = {k:v for k,v in zip(np.arange(len(id_cat)), id_cat)}
    song_rec_df["plylst_id"]
    song_rec_df["song_id"] = song_rec_df["song_id"].map(ix_to_song_id)
    tag_rec_df["tag"] = tag_rec_df["tag"].map(ix_to_tag_id)
    
    result = pd.DataFrame()
    result["id"] = song_rec_df["plylst_id"].map(ix_to_id).drop_duplicates().reset_index(drop = True)
    result["songs"] = song_rec_df.groupby("plylst_id",sort = False).\
        apply(lambda r : [song for song in r["song_id"]]).reset_index(drop = True)
    result["tags"] = tag_rec_df.groupby("plylst_id",sort = False).\
        apply(lambda r : [tag for tag in r["tag"]]).reset_index(drop = True)
    return result

def _recall(pred, true):
    recall = 0.0
    for _, song in enumerate(pred):
        if song in true:
            recall += 1/len(true)
    return recall

def calculate_recall(query, pred, ans):
        
    if set(pred["id"]) != set(ans["id"]):
        raise Exception("결과의 플레이리스트 id가 올바르지 않습니다.")
    
    if set([len(songs) for songs in pred["songs"]]) != \
        set([len(set(songs)) for songs in pred["songs"]]):
        raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")
        
    song_only,song_tag,tag_only,title_only = [],[],[],[]
    for _, row in query.iterrows():
        if not row["tags"] and not row["songs"]:
            title_only.append(row["id"])
            continue
        elif not row["tags"]:
            song_only.append(row["id"])
            continue    
        elif not row["songs"]:
            tag_only.append(row["id"])
            continue
        else:
            song_tag.append(row["id"])
    
    song_recall = [0.0, 0.0, 0.0, 0.0, 0.0]
    tag_recall = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(len(pred)):
        if pred["id"][i] in song_only:    
            song_recall[0] += _recall(pred["songs"][i], ans["songs"][i])/len(song_only)
            tag_recall[0] += _recall(pred["tags"][i], ans["tags"][i])/ len(song_only)
        elif pred["id"][i] in song_tag:    
            song_recall[1] += _recall(pred["songs"][i], ans["songs"][i])/ len(song_tag)
            tag_recall[1] += _recall(pred["tags"][i], ans["tags"][i])/ len(song_tag)
        elif pred["id"][i] in tag_only:    
            song_recall[2] += _recall(pred["songs"][i], ans["songs"][i])/ len(tag_only)
            tag_recall[2] += _recall(pred["tags"][i], ans["tags"][i])/ len(tag_only)
        elif pred["id"][i] in title_only:    
            song_recall[3] += _recall(pred["songs"][i], ans["songs"][i])/ len(title_only)
            tag_recall[3] += _recall(pred["tags"][i], ans["tags"][i])/ len(title_only)
        song_recall[4] += _recall(pred["songs"][i], ans["songs"][i])/ len(pred)
        tag_recall[4] += _recall(pred["tags"][i], ans["tags"][i])/len(pred)

    print(f"Recall song_only | song_tag | tag_only | title_only | total")
    print(f"          {song_recall[0]:.2f}   |   {song_recall[1]:.2f}   |   {song_recall[2]:.2f}   |   {song_recall[3]:.2f}     |   {song_recall[4]:.2f}")
    print(f"          {tag_recall[0]:.2f}   |   {tag_recall[1]:.2f}   |   {tag_recall[2]:.2f}   |   {tag_recall[3]:.2f}     |   {tag_recall[4]:.2f}")
    return song_recall, tag_recall

if __name__ == "__main__":
    sample = 1000
    numsong = [200, 300, 400, 500, 1000]
    numtag = [20, 30, 40, 50, 100]
    with open(stage1_config.SAVE_FOLDER + "model1.pkl","rb") as f:
        model = pkl.load(f)
    with open(stage1_config.STAGE1_PROCESSED_TRAIN_FILE,"rb") as f:
        train = pkl.load(f)
    with open(stage1_config.SAVE_FOLDER + "pipeline1.pkl", "rb") as f:
        pipeline = pkl.load(f)
    for i in range(5):
        query = pd.concat(
                [pd.read_json(f) for f in stage1_config.STAGE1_VALID_Q]
                ).reset_index(drop = True).iloc[:sample]
        test = pd.concat(
                [pd.read_json(f) for f in stage1_config.STAGE1_VALID_A]
                ).reset_index(drop = True).iloc[:sample]
        idx = np.arange(train.shape[0]-len(test),train.shape[0])[:sample]
        pred = model.predict(idx,numsong[i],numtag[i])
        test_argument = recommend(pred, pipeline)
        calculate_recall(query, test_argument, test)
    
    query = pd.concat(
            [pd.read_json(f) for f in stage1_config.STAGE1_VALID_Q]
            ).reset_index(drop = True)
    test = pd.concat(
            [pd.read_json(f) for f in stage1_config.STAGE1_VALID_A]
            ).reset_index(drop = True)
    idx = np.arange(train.shape[0]-len(test),train.shape[0])
    pred = model.predict(idx,1000,100)
    test_argument = recommend(pred, pipeline)
    calculate_recall(query, test_argument, test)
    with open(stage1_config.PROCESSED_DATA_FOLDER + "temp.pkl", "rb") as f:
         temp = pkl.load(f)
    
    
   