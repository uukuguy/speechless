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
