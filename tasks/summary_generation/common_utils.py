import os, json
import pickle
from enum import Enum
from dataclasses import dataclass

# 定义综述类型枚举
class ReviewType(Enum):
    CONCEPT = "concept"  # 技术概念调研
    STATUS = "status"    # 研究现状
    COMPARISON = "comparison"  # 方法对比
    TIMELINE = "timeline"   # 技术脉络

review_desciptions = {
    ReviewType.CONCEPT: "技术概念调研",
    ReviewType.STATUS: "研究现状",
    ReviewType.COMPARISON: "方法对比",
    ReviewType.TIMELINE: "技术脉络"
}

@dataclass
class PaperChunk:
    paper_id: str
    title: str
    chunk_id: str
    content: str


@dataclass
class Citation:
    paper_id: str
    chunk_id: str
    content: str
    title: str

def kb_chunk_to_paper_chunk(kb_chunk):
    if 'entity' in kb_chunk:
        chunk_entity = kb_chunk['entity']
    else:
        chunk_entity = kb_chunk
    paper_chunk = PaperChunk(
        paper_id=chunk_entity["paper_id"],
        title=chunk_entity["paper_title"],
        chunk_id=chunk_entity["chunk_id"],
        content=chunk_entity["chunk_text"]
    )
    return paper_chunk


def cache_or_rebuild(cache_file: str =None): 
    is_pickle = False
    is_json = False
    if cache_file.endswith(".pkl"):
        is_pickle = True
    elif cache_file.endswith(".json"):
        is_json = True
    else:
        raise ValueError("Unknown cache file format")
    def decorator(func):
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_file):
                if is_pickle:
                    return pickle.load(open(cache_file, "rb"))
                elif is_json:
                    return json.load(open(cache_file, "r", encoding='utf-8'))
            else:
                result = func(*args, **kwargs)
                if is_pickle:
                    pickle.dump(result, open(cache_file, "wb"))
                elif is_json:
                    json.dump(result, open(cache_file, "w", encoding='utf-8'), ensure_ascii=False, indent=2, cls=NpEncoder)
                return result
        return wrapper
    return decorator

import re
class LazyDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
            (re.compile(r',(\s*])'), r'\1'),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)

import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
