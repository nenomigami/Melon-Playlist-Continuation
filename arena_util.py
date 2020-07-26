# -*- coding: utf-8 -*-
import io
ArithmeticError
import os
import json
import distutils.dir_util
from collections import Counter

import numpy as np


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, np.generic):#np는 json으로 변환 불가능하므로 base int 로 고쳐준다
            return int(o)
        raise TypeError #if 문에 안걸리면 TypeError

    parent = os.path.dirname(fname) #fname 이 있는 폴더를 리턴
    #distutils.dir_util.mkpath("./arena_data/" + parent) #폴더 한칸 위 아레나데이터/res 를 만든다
    with io.open(fname, "w", encoding="utf8") as f:
        #io.open == open , arena data에서 fname 만들고
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        #json dumping
        f.write(json_str)


def load_json(fname):
    with open(fname, encoding = "utf-8") as f:
        json_obj = json.load(f)

    return json_obj


def debug_json(r):
    print(json.dumps(r, ensure_ascii=False, indent=4))


def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]


def most_popular(playlists, col, topk_count):
    c = Counter()

    for doc in playlists:
        c.update(doc[col])

    topk = c.most_common(topk_count)
    return c, [k for k, v in topk]
