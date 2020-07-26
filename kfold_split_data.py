# -*- coding: utf-8 -*-
"""
Created on Sun May 10 22:47:44 2020

@author: ghdbs
"""

# -*- coding: utf-8 -*-
import copy
import random

import fire
import numpy as np
import pandas as pd
import os

from arena_util import load_json
from arena_util import write_json
from kakao_arena.split_data import ArenaSplitter

"""
K fold 10을 실행하기 위해 데이터마다 폴더를 만들 것 
fold1/train.json , fold1/val.json 이런식으로
1. 인덱스를 고정한다
    1) 인덱스를 편하게 다루기 위해 pandas 사용
    2) 이미 있는 함수를 이용하기 위해 json과도 변환 가능해야함
2. train 데이터에서 index 에 해당하는 데이터를 뽑고 json으로 변환
"""
class KfoldArenaSplitter(ArenaSplitter):
    def __init__(self):
        self.NFOLDS = 5
        self.SEED = 1000
        self.DATA_FOLDER = "." + "/data"
            
    def _generateIdx(self, data):
        """
        Data들의 인덱스를 각 FOLD 로 나눠주는 함수
        ----------
        data : Pandas DataFrame
            나눠질 데이터.
    
        Returns
        -------    
        """
        NFOLDS = self.NFOLDS
        tot = len(data)
        shuffled_idx = np.random.permutation(tot)
        jump = tot//NFOLDS
        for i in range(NFOLDS):    
            te_fold = shuffled_idx[i*jump : (i+1)*jump]
            tr_fold = [item for item in shuffled_idx if item not in te_fold]
            yield tr_fold, te_fold

    def _split_data(self, playlist, tr_idx, te_idx):
        train = playlist.iloc[tr_idx].to_dict("records")
        half = len(te_idx) // 2
        val1 = playlist.iloc[te_idx[:half]].to_dict("records")
        val2 = playlist.iloc[te_idx[half:]].to_dict("records")
        return train, val1, val2

    def run(self, fname):
        
        np.random.seed(self.SEED)
        
        train_org = pd.read_json(fname)
        
        split = self._generateIdx(train_org)
        if self.DATA_FOLDER not in os.listdir("."):
            os.mkdir(self.DATA_FOLDER)
            
        
        for i, (tr_idx, te_idx) in enumerate(split):
            folder = "fold" + str(i) 
            path = self.DATA_FOLDER + "/" + folder
            if folder not in os.listdir(self.DATA_FOLDER):
                os.mkdir(path)    
            
            print("Splitting data...")
            train, val1, val2 = self._split_data(train_org, tr_idx, te_idx)
            
            print(f"fold {i} Original train...")
            write_json(train, path + "/train.json")
            #train.json은 새로 만든다 orig 폴더에
            
            print(f"fold {i} Original val1...")
            write_json(val1, path + "/val1.json")
            
            print(f"fold {i} Original val2...")
            write_json(val2, path + "/val2.json")
            
            print(f"fold {i} Masked val1...")
            val1_q, val1_a = self._mask_data(val1)#validation할것을 마스크해서
            write_json(val1_q, path + "/val1_q.json")
            write_json(val1_a, path + "/val1_a.json")

            print(f"fold {i} Masked val2...")
            val2_q, val2_a = self._mask_data(val2)#validation할것을 마스크해서
            write_json(val2_q, path + "/val2_q.json")
            write_json(val2_a, path + "/val2_a.json")


if __name__ == "__main__":
    fire.Fire(KfoldArenaSplitter)
    #Fire에 ArenaSpitter Class 를 등록
    #python split_data.py run res/train.json
    #클래스의 속성에 접근할 수 있으므로 run 을 실행하고 fname 에 res/train.json 을 넣는다