import os, json
import re
from loguru import logger
from knowledge_base import KnowledgeBase

from common_utils import PaperChunk, kb_chunk_to_paper_chunk

def retrieve_chunk(paper_id: str, chunk_id: int, kb: KnowledgeBase):
    results = kb.query_by_paper_id(paper_id, top_k=chunk_id)
    if len(results) == 0:
        logger.warning(f"Failed to retrieve chunk {chunk_id} from {paper_id}")
        return None
    if len(results) < chunk_id:
        logger.warning(f"No enough chunks ({len(results)}) . Failed to retrieve chunk {chunk_id} from {paper_id}")
        return None

    kb_chunk = results[chunk_id-1]
    # print(f"{kb_chunk=}")
    paper_chunk = kb_chunk_to_paper_chunk(kb_chunk)

    return paper_chunk

def generate_references(citations_file: str, kb: KnowledgeBase):
    with open(citations_file, 'r') as f:
        lines = f.readlines()

    citations = []
    for line in lines:
        line = line.strip()
        found = re.findall(r'^([0-9a-f]{24})-(\d+)', line)
        if found:
            paper_id, citation_id = found[0]
            # logger.debug(f"Found {paper_id=}, {citation_id=}")
            citations.append((paper_id, int(citation_id)))

    citations = sorted(citations)

    references = []
    for citation in citations:
        paper_id, chunk_id = citation
        paper_chunk = retrieve_chunk(paper_id, chunk_id, kb)
        if paper_chunk:
            # ref = f"[{paper_id}-{chunk_id}]: {paper_chunk.title}"
            ref = f"{paper_chunk.title}"
            logger.debug(f"[{paper_id}-{chunk_id}]: {ref}")
            references.append({
                'paper_id': paper_id,
                'chunk_id': chunk_id,
                'ref': ref,
            })
        else:
            logger.warning(f"Failed to retrieve reference for {paper_id}-{citation_id}")
    return references


if __name__ == '__main__':
    kb = KnowledgeBase()
    citations_file = "outputs/损失函数/partial_citations.txt"
    references = generate_references(citations_file, kb)
    print(references)