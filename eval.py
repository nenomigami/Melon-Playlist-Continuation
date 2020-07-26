# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:04:10 2020

@author: ghdbs
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:36:10 2020

@author: ghdbs
"""
from arena_util import load_json
from evaluate2 import Evaluator
import fire

class Eval(Evaluator):
    
    def __init__(self):
        super().__init__()
        
    def _eval(self, gt_fname, rec_fname):
        gt_playlists = load_json(gt_fname) #답지 json 파일을 로드
        gt_dict = {g["id"]: g for g in gt_playlists} # id : playlist 딕셔너리 만듬
        rec_playlists = load_json(rec_fname) #내가 만든 파일 로드

        gt_ids = set([g["id"] for g in gt_playlists]) #id 들 set으로 묶음
        rec_ids = set([r["id"] for r in rec_playlists]) #id 들 set으로 묶음

        if gt_ids != rec_ids:# id들이 같지 않으면 에러
            raise Exception("결과의 플레이리스트 id가 올바르지 않습니다.")

        rec_song_counts = [len(p["songs"]) for p in rec_playlists] #
        rec_tag_counts = [len(p["tags"]) for p in rec_playlists]

        rec_unique_song_counts = [len(set(p["songs"])) for p in rec_playlists]
        rec_unique_tag_counts = [len(set(p["tags"])) for p in rec_playlists]

        if set(rec_unique_song_counts) != set([100]):
            raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")

        if set(rec_unique_tag_counts) != set([10]):
            raise Exception("한 플레이리스트에 중복된 태그 추천은 허용되지 않습니다.")

        music_ndcg = 0.0
        tag_ndcg = 0.0
        recall = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])
            recall += self._recall(gt["songs"], rec["songs"][:100])
        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        recall = recall / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, recall, score            
    
    def evaluate(self, true_fname, pred_fname):
        try:
            music_ndcg, tag_ndcg, recall, score = self._eval(true_fname, pred_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Recall: {recall:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)

from evaluate2 import Evaluator
   
fpred = "./algorithms/ret3.json"
ftrue = "./data/fold0/val_a.json"

eval_ = Eval()
eval_.evaluate(ftrue, fpred)
