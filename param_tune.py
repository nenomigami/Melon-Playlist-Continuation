# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:44:14 2020

@author: ghdbs
"""

import pandas as pd
import numpy as np
from pathlib import Path

#from algorithms.Wrmf import Wrmf 
from WRMF2 import Wrmf 
from pathlib import Path

import os
import json

from evaluate2 import Evaluator


factors = [32*3,32*5,32*7,32*10,32*13]
reg = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
cs = [70,100,150,200,250]
random_state = 777
grid = []
for f in factors:
    for r in reg:
        for c in cs:
            grid.append({"factors" : f, "regularization" : r, "c" : c})

param_list = [grid[i] for i in np.random.randint(0,len(grid),20)]


data_folder = Path("data")
folder = Path("fold" + str(0))
random_state = 1000

ftrain = data_folder / folder / Path("train.json")
fval1 = data_folder / folder / Path("val1_q.json")
fval2 = data_folder / folder / Path("val2_q.json")
ftrain2 = Path("res/test.json")

train = pd.read_json(ftrain)[["id","tags","songs"]]
val1 = pd.read_json(fval1)[["id", "tags","songs"]]
val2 = pd.read_json(fval2)[["id", "tags","songs"]]

train2 = pd.read_json(ftrain2)[["id","tags","songs"]]
train = pd.concat([train,train2]).reset_index(drop=True)
test = val1
#test = pd.concat([val1,val2]).reset_index(drop=True)


fval1_ans = "data/fold0/val1_a.json"
fval2_ans = "data/fold0/val2_a.json"

val1_ans = pd.read_json(fval1_ans)[["id","songs"]]
val2_ans = pd.read_json(fval2_ans)[["id","songs"]]
#val12_ans = pd.concat([val1_ans, val2_ans]).reset_index(drop=True)
val12_ans = pd.read_json(fval1_ans)[["id","songs"]].iloc[:100]
val12_ans.columns = ["playlist_id" , "songs"]
val12_ans = val12_ans.sort_values("playlist_id").reset_index(drop = True)

results = []
param_list = [{'factors': 300, 'regularization': 0.0001, 'c': 100}]
for i, params in enumerate(param_list):
    #wrmf = Wrmf(params, model= "bpr")
    #wrmf.fit(train, test)
    wrmf = Wrmf(params)
    SONG_META = Path("res/song_meta.json")
    song_meta = pd.read_json(SONG_META)
    processed_total = wrmf.preprocess(train, test[:100], song_meta)
    wrmf.fit(processed_total)
    wrmf.n_train
    test_pred,_ = wrmf.recommend(processed_total, 200, 100)
    #test_pred = wrmf.recommend(test, 200, 100)
    test_pred
    #test_pred_eval = test_pred.groupby("playlist_id").\
    #                            apply(lambda grp : [id for id in grp["song_id"]]).reset_index(0)
    test_pred_eval = test_pred.groupby("plylst_id").\
                                apply(lambda grp : [id for id in grp["song_id"]]).reset_index(0)
    test_pred_eval.columns = ["playlist_id" , "songs"]

    val1_pred = test_pred_eval[:len(val1)]
    val2_pred = test_pred_eval[len(val1):]
    
    val12_pred = pd.concat([val1_pred, val2_pred]).reset_index(drop = True)
    val12_pred.columns = ["playlist_id" , "songs"]

    
    if set(val12_pred["playlist_id"]) != set(val12_ans["playlist_id"]):
        raise Exception("결과의 플레이리스트 id가 올바르지 않습니다.")
    
    if set([len(songs) for songs in val12_pred["songs"]]) != \
        set([len(set(songs)) for songs in val12_pred["songs"]]):
        raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")
                
    def _recall(pred, true):
        recall = 0.0
        for _, song in enumerate(pred):
            if song in true:
                recall += 1/len(true)
        return recall
    song_only,song_tag,tag_only,title_only = [],[],[],[]
    for _, row in test.iterrows():
        if not row["tags"] and not row["songs"]:
            title_only.append(row["id"])
            continue
        if not row["tags"]:
            tag_only.append(row["id"])
            continue
        if not row["songs"]:
            song_only.append(row["id"])
            continue
        else:
            song_tag.append(row["id"])
    
    recall = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(len(val12_pred)):
        if val12_pred["playlist_id"][i] in song_only:    
            recall[0] += _recall(val12_pred["songs"][i], val12_ans["songs"][i])
        elif val12_pred["playlist_id"][i] in song_tag:    
            recall[1] += _recall(val12_pred["songs"][i], val12_ans["songs"][i])
        elif val12_pred["playlist_id"][i] in tag_only:    
            recall[2] += _recall(val12_pred["songs"][i], val12_ans["songs"][i])
        elif val12_pred["playlist_id"][i] in title_only:    
            recall[3] += _recall(val12_pred["songs"][i], val12_ans["songs"][i])
        recall[4] += _recall(val12_pred["songs"][i], val12_ans["songs"][i])
    recall[0] = recall[0] / len(song_only)
    recall[1] = recall[1] / len(song_tag)
    recall[2] = recall[2] / len(tag_only)
    recall[3] = recall[3] / len(title_only)
    recall[4] = recall[4] / len(val12_pred)
    results.append([i, params, recall])



wrmf.model.user_factors.shape
wrmf.model.item_factors.shape
wrmf.song_dict
wrmf.tag_dict


from evaluate2 import Evaluator

import json
result = pd.DataFrame()
result["id"] = test_pred["playlist_id"].drop_duplicates().reset_index(drop=True)
result["songs"] = test_pred.groupby("playlist_id").apply(lambda grp: 
                                                         [id for id in grp["song_id"]]).reset_index(drop = True)
result = result.to_dict("records")
fresult = "results200.json"
with open(fresult, 'w', encoding='utf-8') as f:
    f.write(json.dumps(result, ensure_ascii=False))

fpred = fresult
ftrue = "./data/fold0/val1_a.json"

eval_ = Eval()
eval_.evaluate(ftrue, fpred)

#결과저장폴더 = Path()
#결과저장폴더 results0~9.json
#평균 eval들 보고 best

for param in grid
    for folder in os.listdir(data_folder):
        folder = Path(folder)
        ftrain = data_folder / folder / Path("train.json")
        ftrue = data_folder / folder / Path("val_a.json")
    
        train = pd.read_json(ftrain)[["id","tags","songs"]]
        test = pd.read_json(fval)[["id","tags","songs"]]
    
        
        
        wrmf = WRMF(params, tags = True)
        wrmf.fit(train)
        item_ret, tag_ret = wrmf.predict(test)

        te_ids = test.index
        returnval = []
        for _id, rec, tag_rec in zip(te_ids, item_ret, tag_ret):
            returnval.append({"id": _id,
                               "songs": rec[:100],
                               "tags": tag_rec[:10]
                               })


