# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:36:10 2020

@author: ghdbs
"""
from arena_util import load_json
from kakao_arena.evaluate import ArenaEvaluator
import fire

class Evaluator(ArenaEvaluator):
    
    def __init__(self):
        super().__init__()
        
    def _recall(self, gt, rec):
        """
        recall 값을 계산
        예측한 곡들 중 gt에 있는 노래 수 / gt에 있는 전체 노래 수
        => 정확하지 않은 노래를 많이 답으로 제시할 수록 리콜은 감소,
        즉 순서를 따지기 전에 정확한 노래부터 학습 필요하기 때문에 리콜을 최대화
        Parameters
        ----------
        gt : list
            answer.
        rec : list
            result to be checked.
        """
        recall = 0.0
        for _, song in enumerate(gt):
            if song in rec:
                recall += 1/len(gt)
                
        return recall
    
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

        if set(rec_song_counts) != set([100]): #무조건 답지는 
            raise Exception("추천 곡 결과의 개수가 맞지 않습니다.")

        if set(rec_tag_counts) != set([10]):
            raise Exception("추천 태그 결과의 개수가 맞지 않습니다.")

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
            
if __name__ == "__main__":
    fire.Fire(Evaluator)
