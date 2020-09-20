# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:21:49 2020

@author: ghdbs
"""
import sys
sys.path.insert(1,'../')
import stage1_config
import pandas as pd
import numpy as np
import copy 
class DataGenerator:
    """
    1. train을 27 : 3 : 1 로 나눈다.
    2. train2, train3 을 test 와 같은 모양으로 masking 한다
    3. stage1 에 쓰일 train, valid 를 한 파일로 만든다.
    """
    
    def _mask(self, playlists, mask_cols, del_cols):
        np.random.seed(1000)
        playlists = playlists.reset_index(drop=True)
        q_pl = copy.deepcopy(playlists) #playlist 깊은 복사 2개
        a_pl = copy.deepcopy(playlists)

        for i in range(len(playlists)):
            #플레이 리스트를 돌면서
            for del_col in del_cols:
                q_pl.at[i,del_col] = []
                #q_pl (플레이리스트) 의 i번째 행의 del_col 키값은 []로 만든다
                if del_col == 'songs':
                    #삭제하는 col이 songs이면 
                    a_pl.at[i,del_col] = a_pl.loc[i,del_col][:100]
                    #a_pl의 songs는 col은 100개까지로 줄인다
                elif del_col == 'tags':
                    #삭제하려는 col이 tags면
                    a_pl.at[i,del_col] = a_pl.loc[i,del_col][:10]
                    #a_pl의 tag는 10개까지로 줄인다

            for col in mask_cols:
                #마스크 하려는 컬럼을 돌면서
                mask_len = len(playlists.loc[i,col])
                #마스크 하려는 컬럼의 키값에 해당하는 밸류의 길이를 재고
                mask = np.full(mask_len, False)
                #mask len만큼 false로 채운 리스트를 두고
                mask[:mask_len//2] = True
                #절반은 True로 채우고
                np.random.shuffle(mask)
                #mask를 섞는다

                q_pl.at[i,col] = list(np.array(q_pl.loc[i,col])[mask])
                #q_pl 의 마스크하려는 컬럼은 mask True, False 에 해당하는 데이터만 뽑은 걸로 교체
                a_pl.at[i,col] = list(np.array(a_pl.loc[i,col])[np.invert(mask)])
                #a_pl의 마스크 하려는 컬럼은 mask의 반대 False, True에 해당하는 데이터만 뽑은걸로 교체
        return q_pl, a_pl
    
    def _mask_data(self, playlists):
        playlists = copy.deepcopy(playlists)
        tot = len(playlists)
        song_only = playlists.iloc[:int(tot * 0.42)]
        song_and_tags = playlists.iloc[int(tot * 0.42):int(tot * 0.81)]
        tags_only = playlists.iloc[int(tot * 0.81):int(tot * 0.92)]
        title_only = playlists.iloc[int(tot * 0.98):]

        print(f"Total: {len(playlists)}, "
              f"Song only: {len(song_only)}, "
              f"Song & Tags: {len(song_and_tags)}, "
              f"Tags only: {len(tags_only)}, "
              f"Title only: {len(title_only)}")
        
        song_q, song_a = self._mask(song_only, ['songs'], ['tags'])
        songtag_q, songtag_a = self._mask(song_and_tags, ['songs', 'tags'], [])
        tag_q, tag_a = self._mask(tags_only, ['tags'], ['songs'])
        title_q, title_a = self._mask(title_only, [], ['songs', 'tags'])
        q = pd.concat([song_q, songtag_q, tag_q, title_q]).reset_index(drop=True)
        a = pd.concat([song_a, songtag_a, tag_a, title_a]).reset_index(drop=True)

        shuffle_indices = np.arange(len(q))
        np.random.shuffle(shuffle_indices)

        q = q.iloc[shuffle_indices].reset_index(drop=True)
        a = a.iloc[shuffle_indices].reset_index(drop=True)

        return q, a  
    
    def _split(self, train_file):
        train = pd.read_json(train_file)
        length = len(train)
        train1 = train.iloc[:int(length * 27/31)]  
        train2 = train.iloc[int(length * 27/31):int(length * 30/31)]
        train3 = train.iloc[int(length * 30/31):]
        return train1, train2, train3    

    def split_data(self, train, write = False):
        train1, train2, train3 = self._split(train)
        train2_q, train2_a = self._mask_data(train2)
        train3_q, train3_a = self._mask_data(train3)
        if write == True:
            train1.to_json(stage1_config.SPLITED_TRAINING_DATA_FILE[0])
            train2_q.to_json(stage1_config.SPLITED_TRAINING_DATA_FILE[1])
            train2_a.to_json(stage1_config.SPLITED_TRAINING_DATA_FILE[2])
            train3_q.to_json(stage1_config.SPLITED_TRAINING_DATA_FILE[3])
            train3_a.to_json(stage1_config.SPLITED_TRAINING_DATA_FILE[4])
        return train1, train2_q, train2_a, train3_q, train3_a

if __name__ == "__main__":
    DataGenerator().split_data(stage1_config.TRAINING_DATA_FILE)
