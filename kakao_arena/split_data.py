# -*- coding: utf-8 -*-
import copy
import random

import fire
import numpy as np

from arena_util import load_json
from arena_util import write_json


class ArenaSplitter:
    def _split_data(self, playlists):
        tot = len(playlists) #playlist 의 길이
        train = playlists[:int(tot*0.90)] #총 길이의 80퍼,
        val = playlists[int(tot*0.90):] #총 길이의 20퍼

        return train, val

    def _mask(self, playlists, mask_cols, del_cols):
        """
        ----------
        playlists : list
        mask_cols : 
        del_cols : 리스트, 삭제할 컬럼
        Returns
        -------
        q_pl : TYPE
            DESCRIPTION.
        a_pl : TYPE
            DESCRIPTION.

        """
        q_pl = copy.deepcopy(playlists) #playlist 깊은 복사 2개
        a_pl = copy.deepcopy(playlists)

        for i in range(len(playlists)):
            #플레이 리스트를 돌면서
            for del_col in del_cols:
                q_pl[i][del_col] = []
                #q_pl (플레이리스트) 의 i번째 행의 del_col 키값은 []로 만든다
                if del_col == 'songs':
                    #삭제하는 col이 songs이면 
                    a_pl[i][del_col] = a_pl[i][del_col][:100]
                    #a_pl의 songs는 col은 100개까지로 줄인다
                elif del_col == 'tags':
                    #삭제하려는 col이 tags면
                    a_pl[i][del_col] = a_pl[i][del_col][:10]
                    #a_pl의 tag는 10개까지로 줄인다

            for col in mask_cols:
                #마스크 하려는 컬럼을 돌면서
                mask_len = len(playlists[i][col])
                #마스크 하려는 컬럼의 키값에 해당하는 밸류의 길이를 재고
                mask = np.full(mask_len, False)
                #mask len만큼 false로 채운 리스트를 두고
                mask[:mask_len//2] = True
                #절반은 True로 채우고
                np.random.shuffle(mask)
                #mask를 섞는다

                q_pl[i][col] = list(np.array(q_pl[i][col])[mask])
                #q_pl 의 마스크하려는 컬럼은 mask True, False 에 해당하는 데이터만 뽑은 걸로 교체
                a_pl[i][col] = list(np.array(a_pl[i][col])[np.invert(mask)])
                #a_pl의 마스크 하려는 컬럼은 mask의 반대 False, True에 해당하는 데이터만 뽑은걸로 교체
        return q_pl, a_pl

    def _mask_data(self, playlists):
        playlists = copy.deepcopy(playlists)
        #playlist 리스트 딥카피
        tot = len(playlists)
        #플레이리스트 총 수
        song_only = playlists[:int(tot * 0.3)]
        #0.3 / 0.5 / 0.15 / 0.05
        #곡, 곡태그, 태그만, 제목만
        song_and_tags = playlists[int(tot * 0.3):int(tot * 0.8)]
        tags_only = playlists[int(tot * 0.8):int(tot * 0.95)]
        title_only = playlists[int(tot * 0.95):]

        print(f"Total: {len(playlists)}, "
              f"Song only: {len(song_only)}, "
              f"Song & Tags: {len(song_and_tags)}, "
              f"Tags only: {len(tags_only)}, "
              f"Title only: {len(title_only)}")

        song_q, song_a = self._mask(song_only, ['songs'], ['tags'])
        #tag를 삭제하고 song 만 남기는데 송 절반만 뽑는다
        #song_q에는 송만 50%들어있고 song_a에는 q에 없는 송과 tag 10개가 들어있음
        songtag_q, songtag_a = self._mask(song_and_tags, ['songs', 'tags'], [])
        #songtag_q 에는 song 50퍼, 태그 50퍼, songtag_a 에는 q에는 없는 송과 tag 전체(최대 10개)
        tag_q, tag_a = self._mask(tags_only, ['tags'], ['songs'])
        #태그를 50퍼 분배, tag_a는 노래 100개만 
        title_q, title_a = self._mask(title_only, [], ['songs', 'tags'])
        #타이틀만 하고 a에는 song 100개 태그 10개까지만
        q = song_q + songtag_q + tag_q + title_q
        a = song_a + songtag_a + tag_a + title_a

        shuffle_indices = np.arange(len(q))
        np.random.shuffle(shuffle_indices)

        q = list(np.array(q)[shuffle_indices])
        a = list(np.array(a)[shuffle_indices])

        return q, a #셔플하고 주기

    def run(self, fname):
        random.seed(777) 

        print("Reading data...\n")
        playlists = load_json(fname)
        #playlist는 태그, id, title, 곡들, 좋아요, 업데이트 날짜가 들어있는 딕셔너리 리스트
        random.shuffle(playlists)
        print(f"Total playlists: {len(playlists)}")

        print("Splitting data...")
        train, val = self._split_data(playlists)
        #플레이리스트 나누기

        print("Original train...")
        write_json(train, "orig/train.json")
        #train.json은 새로 만든다 orig 폴더에
        print("Original val...")
        write_json(val, "orig/val.json")

        print("Masked val...")
        val_q, val_a = self._mask_data(val)#validation할것을 마스크해서
        write_json(val_q, "questions/val.json")#q는 퀘스쳔 폴더, a는 앤서 폴더에 넣기
        write_json(val_a, "answers/val.json")


if __name__ == "__main__":
    fire.Fire(ArenaSplitter)
    #Fire에 ArenaSpitter Class 를 등록
    #python split_data.py run res/train.json
    #클래스의 속성에 접근할 수 있으므로 run 을 실행하고 fname 에 res/train.json 을 넣는다