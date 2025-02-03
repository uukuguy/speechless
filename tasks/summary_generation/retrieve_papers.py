#!/usrbin/env python3
"""
主任务是基于检索增强的多智能协作的结构化长文献综述生成，能够响应不同类型的综述指令，：
  a.对一个技术概念的调研综述；
  b.对一个方向研究现状的综述；
  c.对多个方法的对比分析综述；
  d.对一个技术方法研究脉络的综述。
论文已经切片保存在文献知识库（标准的向量数据库）中。
第一步需要将用户提问转换成能从文献知识库中召回尽可能丰富、准确的论文内容切片的向量数据库检索查询。请生成满足以上要求的检索查询内容。

# ---------- DeepSeek Reasoner ----------

根据不同的综述任务类型和用户输入，生成的检索查询应包含以下分层结构内容（示例模板）：

Ⅰ. 技术概念调研检索模板
【核心概念】用户输入的技术概念术语（扩展同义词）
【知识维度】"定义解释" OR "理论基础"
OR "核心原理"
OR "技术演进"
AND ("应用场景"
OR "典型用例"
OR "局限性")
【时间范围】根据概念成熟度选择：
• 新兴概念：最近3年
• 成熟概念：无限制

Ⅱ. 研究现状综述检索模板
【研究主题】用户输入研究方向 +
("state-of-the-art" OR "research progress"
OR "current challenges"
OR "future trends")
【关键要素】("survey" OR "review")
NEAR/5 ("comparative analysis"
OR "experimental evaluation"
OR "benchmark results")
【时间限定】最近5年（前沿领域可缩至3年）

Ⅲ. 多方法对比分析检索模板
【方法集合】用户指定方法列表 +
("comparative study"
OR "performance comparison"
OR "advantage/disadvantage analysis")
【对比维度】"accuracy metrics"
OR "computational complexity"
OR "scalability"
OR "implementation cost"
（根据领域添加特有指标）
【证据类型】优先召回包含对比表格/实验数据/统计分析的片段

Ⅳ. 技术脉络分析检索模板
【技术发展】"historical evolution"
OR "milestone breakthroughs"
OR "paradigm shift"
【关联关系】"technical lineage"
NEAR/3 ("theoretical foundation"
OR "algorithm improvement"
OR "architecture innovation")
【时间轨迹】按十年为阶段分批次检索：
• 基础理论阶段（最早文献）
• 核心突破阶段（关键年份）
• 最新发展（近3年）

通用增强策略：

同义词扩展：使用领域本体库扩展技术术语
否定排除：添加 -"patent" -"news" 过滤非论文内容
层次检索：先召回高被引论文片段，再补充最新顶会论文
多语言处理：对非英语文献增加术语的英文原词映射
示例应用：
当用户查询"对比分析Transformer与CNN在图像识别中的表现"时，生成的检索式应为：
("Transformer architecture" OR "Self-attention mechanism")
AND ("CNN" OR "Convolutional Neural Networks")
AND ("image recognition benchmark"
OR "comparative performance analysis"
OR "computational efficiency comparison")
NEAR/5 ("accuracy metrics"
OR "FLOPs"
OR "training convergence")
YEAR:2017-2023

这种结构化检索策略相比简单关键词搜索，可使召回准确率提升37%（基于ACL 2023检索增强论文实验数据）

"""
import os, json
import pickle
from typing import List
from common_utils import cache_or_rebuild
from common_utils import ReviewType, PaperChunk, Citation, review_type_descriptions, kb_chunk_to_paper_chunk 
from llm_utils import LLMClient
from intent_recognition import classify_intent_with_prompt
from knowledge_base import KnowledgeBase 
from loguru import logger
from tqdm import tqdm

verbose=True

class QueryExpander:
    """查询扩展智能体,负责将用户简短查询扩展为更全面的检索关键词"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def expand_query(self, query: str, review_type: ReviewType) -> List[str]:
        """扩展用户查询为多个相关查询"""
        prompt = f"""请基于以下用户查询,生成5-8个相关的检索关键词或短语,以帮助全面检索相关文献:
        用户查询: {query}
        综述类型: {review_type_descriptions[review_type]}
        要求:
        1. 关键词应该覆盖该主题的不同方面
        2. 包含该领域的专业术语
        3. 考虑近义词和相关概念
        4. 以JSON列表格式返回
        """
        
        generated_text = self.llm.generate(prompt, verbose=verbose)
        return json.loads(generated_text)

class ContentRetriever:
    """内容检索智能体,负责从文献库检索相关内容"""
    
    def __init__(self, llm_client, chunk_size=1000):
        self.llm = llm_client
        self.chunk_size = chunk_size

        self.kb = KnowledgeBase()
    
    def retrieve_by_queries(self, queries: List[str], min_papers=50, max_chunks=1000) -> List[PaperChunk]:
        """基于多个查询检索文献内容"""
        all_chunks = []
        papers_seen = set()
        
        for query in queries:
            chunks = self.kb.search_papers(query)
            for chunk in chunks:
                if verbose:
                    logger.debug(f"Retrieved chunk: {chunk}")

                chunk = kb_chunk_to_paper_chunk(chunk)

                if chunk.paper_id not in papers_seen:
                    all_chunks.append(chunk)
                    papers_seen.add(chunk.paper_id)
                    
            if len(papers_seen) >= min_papers:
                break
                
        return all_chunks

    def retrieve_papers_by_queries(self, queries: List[str], min_papers=50) -> List[str]:
        """基于多个查询检索文献内容"""
        papers_seen = set()
        
        for query in queries:
            chunks = self.kb.search_papers(query, top_k=1000)
            for chunk in chunks:
                # if verbose:
                #     logger.debug(f"Retrieved chunk: {chunk}")

                chunk = kb_chunk_to_paper_chunk(chunk)

                if chunk.paper_id not in papers_seen:
                    papers_seen.add(chunk.paper_id)
                    logger.info(f"Retrieved paper: {chunk.paper_id}: {chunk.title}")
                    if len(papers_seen) >= min_papers:
                        break
                    
            if len(papers_seen) >= min_papers:
                break
                
        return papers_seen
    

from common_utils import Paper
def generate_papers_content(query: str, output_file: str):
    root_dir = f"outputs/{query}"
    paper_chunks_file = f"{root_dir}/paper_chunks.pkl"
    with open(paper_chunks_file, 'rb') as f:
        paper_chunks = pickle.load(f)
    
    papers = {}
    for chunk in tqdm(paper_chunks, desc="Generating papers content"):
        if chunk.paper_id not in papers:
            papers[chunk.paper_id] = Paper(paper_id=chunk.paper_id, title=chunk.title)
        papers[chunk.paper_id].add_chunk(chunk)
    for paper in papers.values():
        paper.post_process() 

    with open(output_file, 'wb') as f:
        pickle.dump(papers, f)

    logger.info(f"Generated {len(papers)} papers content for query '{query}' to {output_file}.")

def do_retrieve_papers(query):
    """
    执行检索论文内容的任务
    """

    model_name = os.getenv("OPENAI_DEFAULT_MODEL")
    llm_client = LLMClient(model_name=model_name)

    intent_type = classify_intent_with_prompt(llm_client, query)
    # assert intent_type == args.review_type
    review_type = intent_type
    # review_type = args.review_type

    root_dir = f"outputs/{query}"
    os.makedirs(root_dir, exist_ok=True)

    # 1. 扩展查询
    query_expander = QueryExpander(llm_client)
    @cache_or_rebuild(cache_file=f"{root_dir}/expanded_queries.json")
    def do_expand_query(query: str, review_type: ReviewType) -> List[str]:
        """扩展查询"""
        return query_expander.expand_query(query, review_type)
    expanded_queries = do_expand_query(query, review_type)
    
    # 2. 检索内容

    retriever = ContentRetriever(llm_client)

    def retrieve_chunks(queries: List[str], min_papers=50) -> List[PaperChunk]:
        chunks = retriever.retrieve_by_queries(queries)
        logger.info(f"Retrieved {len(chunks)} chunks")
        if len(chunks) < min_papers:
            print(f"Warning: Only found {len(chunks)} relevant chunks, less than minimum requirement of {min_papers}")
        
        return chunks

    def retrieve_papers(queries: List[str], min_papers=50) -> List[str]:
        papers_seen = retriever.retrieve_papers_by_queries(queries)
        logger.info(f"Retrieved {len(papers_seen)} papers")
        if len(papers_seen) < min_papers:
            print(f"Warning: Only found {len(papers_seen)} relevant papers, less than minimum requirement of {min_papers}")
        
        return papers_seen

    @cache_or_rebuild(cache_file=f"{root_dir}/papers_seen.pkl")
    def do_retrieve_papers(expanded_queries: List[str], min_papers=50) -> List[str]:
        return retrieve_papers(expanded_queries, min_papers=min_papers)
    papers_seen = do_retrieve_papers(expanded_queries, min_papers=50)
    print(f"{papers_seen=}")

    def retrieve_papers_content(papers_seen: List[str]) -> List[PaperChunk]:
        all_chunks = []
        for paper_id in tqdm(papers_seen, desc="Retrieving papers content"):
            paper_chunks = retriever.kb.query_by_paper_id(paper_id, top_k=1000)
            paper_chunks = [kb_chunk_to_paper_chunk(chunk) for chunk in paper_chunks]
            all_chunks.extend(paper_chunks)
        return all_chunks
    @cache_or_rebuild(cache_file=f"{root_dir}/paper_chunks.pkl")
    def do_retrieve_papers_content(papers_seen: List[str]) -> List[PaperChunk]:
        return retrieve_papers_content(papers_seen)
    paper_chunks = do_retrieve_papers_content(papers_seen)
    print(f"{len(paper_chunks)=}")

    # 3. 生成论文内容
    output_file = f"{root_dir}/papers_content.pkl"
    generate_papers_content(query, output_file)



def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--do_generate_papers_content", action="store_true", help="Generate papers content")
    args = parser.parse_args()
    return args
    
def main():
    args = get_args()
    if args.do_generate_papers_content:
        generate_papers_content(args.query, f"outputs/{args.query}/papers_content.pkl")
    else:
        do_retrieve_papers(args.query)

if __name__ == "__main__":
    main()

