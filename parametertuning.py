# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 23:44:14 2020

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

DATA_FOLDER = Path("data/fold0")

FTRAIN = DATA_FOLDER  / Path("train.json")
FVALID1 = DATA_FOLDER / Path("val1_q.json")
FVALID2 = DATA_FOLDER / Path("val2_q.json")
FTEST = Path("res/test.json")

FVALID1_A = DATA_FOLDER / Path("val1_a.json")
FVALID2_A = DATA_FOLDER / Path("val2_a.json")
SONG_META = Path("res/song_meta.json")

stage1_train = pd.concat([pd.read_json(FTRAIN),pd.read_json(FTEST)]).reset_index(drop=True)
stage1_valid = pd.read_json(FVALID1)
stage1_valid_a = pd.read_json(FVALID1_A)[["id","tags","songs"]]
song_meta = pd.read_json(SONG_META)
results = []
for i, params in enumerate(param_list):
    model = Wrmf(params)
    processed_total = model.preprocess(stage1_train, stage1_valid, song_meta)
    model.fit(processed_total)
    
    song_rec_df, tag_rec_df = model.recommend(processed_total, 200, 100)
    
    def to_result(song_uir_matrix, tag_uir_matrix):
        import json
        result = pd.DataFrame()
        result["id"] = song_uir_matrix["plylst_id"].drop_duplicates().sort_values().reset_index(drop = True)
        result["songs"] = song_uir_matrix.groupby("plylst_id", sort=False).apply(lambda grp: 
                                                                     [id for id in grp["song_id"]]).reset_index(drop = True)
        result["tags"] = tag_uir_matrix.groupby("plylst_id", sort=False).apply(lambda grp: 
                                                                     [id for id in grp["tag"]]).reset_index(drop = True)    
        return result
    
    result = to_result(song_rec_df, tag_rec_df)
    from util import calculate_song_recall
    results.append([params, calculate_song_recall(stage1_valid, result, stage1_valid_a)])

fresult = "튜닝2.txt"
with open(fresult, 'w', encoding='utf-8') as f:
    f.write(json.dumps(results, ensure_ascii=False))
