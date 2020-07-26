# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:30:36 2020

@author: ghdbs
"""

import numpy as np
import pandas as pd


def list2multirows(df, target, colname):
    temp = pd.DataFrame(df[target].tolist(), index = df.index).stack()
    temp = temp.reset_index(0)
    temp.columns = colname
    return temp

def map_cols(df, key_col, dict):
    """
    dictionary의 해당 키에 해당하는 value값으로 columns을 만든다
    """
    df = df.copy()
    df[key_col] = df[key_col].map(dict.get)
    return df

def makeDict(song_list, start_idx):
    song_dict = {}    
    idx = start_idx
    for songs in song_list:
        for song in songs:         
            if song not in song_dict:
               song_dict[song] = idx
               idx += 1
    return song_dict

def calculate_song_recall(query, pred, ans):
        
    if set(pred["id"]) != set(ans["id"]):
        raise Exception("결과의 플레이리스트 id가 올바르지 않습니다.")
    
    if set([len(songs) for songs in pred["songs"]]) != \
        set([len(set(songs)) for songs in pred["songs"]]):
        raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")
    
    def _recall(pred, true):
        recall = 0.0
        for _, song in enumerate(pred):
            if song in true:
                recall += 1/len(true)
        return recall
        
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
    for i in range(len(pred)):
        if pred["id"][i] in song_only:    
            song_recall[0] += _recall(pred["songs"][i], ans["songs"][i])
        elif pred["id"][i] in song_tag:    
            song_recall[1] += _recall(pred["songs"][i], ans["songs"][i])
        elif pred["id"][i] in tag_only:    
            song_recall[2] += _recall(pred["songs"][i], ans["songs"][i])
        elif pred["id"][i] in title_only:    
            song_recall[3] += _recall(pred["songs"][i], ans["songs"][i])
        song_recall[4] += _recall(pred["songs"][i], ans["songs"][i])
    
    song_recall[0] /= len(song_only)
    song_recall[1] /= len(song_tag)
    song_recall[2] /= len(tag_only)
    song_recall[3] /= len(title_only)
    song_recall[4] /= len(pred)
    
    tag_recall = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(len(pred)):
        if pred["id"][i] in song_only:    
            tag_recall[0] += _recall(pred["tags"][i], ans["tags"][i])
        elif pred["id"][i] in song_tag:    
            tag_recall[1] += _recall(pred["tags"][i], ans["tags"][i])
        elif pred["id"][i] in tag_only:    
            tag_recall[2] += _recall(pred["tags"][i], ans["tags"][i])
        elif pred["id"][i] in title_only:    
            tag_recall[3] += _recall(pred["tags"][i], ans["tags"][i])
        tag_recall[4] += _recall(pred["tags"][i], ans["tags"][i])

    tag_recall[0] /= len(song_only)
    tag_recall[1] /= len(song_tag)
    tag_recall[2] /= len(tag_only)
    tag_recall[3] /= len(title_only)
    tag_recall[4] /= len(pred)
        
    return song_recall, tag_recall
