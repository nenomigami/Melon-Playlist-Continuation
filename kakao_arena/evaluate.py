# -*- coding: utf-8 -*-
import fire
import numpy as np

from arena_util import load_json


class ArenaEvaluator:
    
    def _idcg(self, l):
        #이상적인 dcg인 idcg 계산
        #여기서는 relevance 를 그냥 바이너리로 0 ,1 사용하는듯
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]
        #100 까지의 idcg

    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:#gt 답지
                dcg += 1.0 / np.log(i + 2) #일단 gt 안에 있으면 relevance 1
                #그리고 어디에 붙어있는지에 따라 할인율 달라짐
#궁금증 : 중간에 막 곡들 있으면??
        return dcg / self._idcgs[len(gt)]

    def _eval(self, gt_fname, rec_fname):
        gt_playlists = load_json(gt_fname) #답지 json 파일을 로드
        gt_dict = {g["id"]: g for g in gt_playlists} # id : playlist 딕셔너리 만듬
        rec_playlists = load_json(rec_fname) #내가 만든 파일 로드

        gt_ids = set([g["id"] for g in gt_playlists]) #id 들 set으로 묶음
        rec_ids = set([r["id"] for r in rec_playlists]) #id 들 set으로 묶음

        if gt_ids != rec_ids:# id들이 같지 않으면 에러
            raise Exception("결과의 플레이리스트 수가 올바르지 않습니다.")

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

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, gt_fname, rec_fname):
        try:
            music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    fire.Fire(ArenaEvaluator)
