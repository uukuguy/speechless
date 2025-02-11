
import os, json
import pickle
from typing import Dict, List, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from loguru import logger
from textwrap import dedent

from zhipuai import ZhipuAI
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from knowledge_base import KnowledgeBase 
kb = KnowledgeBase()

from common_utils import ReviewType, PaperChunk, Citation, review_type_descriptions, kb_chunk_to_paper_chunk 
from citation_utils import generate_references
from llm_utils import LLMClient

verbose = True

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


import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# class QueryExpander:
#     """查询扩展器：扩充原始查询以提高检索覆盖度"""
#     def __init__(self):
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words = set(stopwords.words('english'))
        
#     def expand_query(self, query: str) -> List[str]:
#         """扩展原始查询"""
#         expanded_queries = [query]
        
#         # # 1. 分词和预处理
#         # tokens = word_tokenize(query.lower())
#         # tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
#         #          if token not in self.stop_words and token.isalnum()]
        
#         # # 2. 生成关键词组合
#         # for i in range(len(tokens)):
#         #     # 单个关键词
#         #     expanded_queries.append(tokens[i])
#         #     # 相邻词对
#         #     if i < len(tokens) - 1:
#         #         expanded_queries.append(f"{tokens[i]} {tokens[i+1]}")
                
#         # # 3. 添加领域特定扩展
#         # # 根据不同综述类型添加相关术语
#         # if "research status" in query.lower() or "现状" in query:
#         #     expanded_queries.extend(["current progress", "recent advances", "challenges"])
#         # elif "development" in query.lower() or "发展" in query:
#         #     expanded_queries.extend(["evolution", "timeline", "milestone"])
#         # elif "comparison" in query.lower() or "对比" in query:
#         #     expanded_queries.extend(["comparison", "versus", "advantages"])
            
#         return list(set(expanded_queries))

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
    
    def retrieve_by_queries(self, queries: List[str], min_papers=50, max_chunks=1000) -> List[PaperChunk]:
        """基于多个查询检索文献内容"""
        all_chunks = []
        papers_seen = set()
        
        for query in queries:
            chunks = kb.search_papers(query)
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
    
    def cluster_chunks(self, chunks: List[PaperChunk]) -> Dict[str, List[PaperChunk]]:
        """将检索到的内容按主题聚类
        
        使用LLM为每个chunk生成主题标签,然后根据标签相似度聚类
        """
        # 1. 为每个chunk生成主题标签
        chunk_topics = {}
        bad_chunks = []
        for chunk in chunks:
            prompt = f"""请为以下文献片段生成3个主题标签,以逗号分隔:
            标题: {chunk.title}
            内容: {chunk.content[:500]}  # 使用前500字符以节省token
            """
            response = self.llm.generate(prompt, verbos=verbose)
            if response.startswith("error"):
                bad_chunks.append(chunk)
                continue
            topics = self.llm.generate(prompt, verbose=verbose).strip().split(",")
            chunk_topics[chunk.chunk_id] = [t.strip() for t in topics]
        for c in bad_chunks:
            chunks.remove(c)
        
        # 2. 计算chunk间的主题相似度
        from collections import defaultdict
        topic_clusters = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            topics_i = set(chunk_topics[chunk.chunk_id])
            max_similarity = 0
            best_cluster = None
            
            # 寻找最相似的已有簇
            for topic, cluster in topic_clusters.items():
                if not cluster:  # 跳过空簇
                    continue
                # 使用第一个chunk的主题作为簇的代表
                cluster_topics = set(chunk_topics[cluster[0].chunk_id])
                similarity = len(topics_i & cluster_topics) / len(topics_i | cluster_topics)
                
                if similarity > max_similarity and similarity > 0.3:  # 相似度阈值
                    max_similarity = similarity
                    best_cluster = topic
            
            # 如果找到合适的簇就加入,否则创建新簇
            if best_cluster is not None:
                topic_clusters[best_cluster].append(chunk)
            else:
                # 使用当前chunk的第一个主题作为新簇的标识
                topic_clusters[chunk_topics[chunk.chunk_id][0]].append(chunk)
        
        return dict(topic_clusters)


class OutlineGenerator:
    """大纲生成智能体,负责生成综述框架"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def generate_outline(self, query: str, review_type: ReviewType, 
                        sample_chunks: List[PaperChunk]) -> List[str]:
        """生成综述大纲"""
        # 构建输入上下文
        context = self._prepare_context(query, review_type, sample_chunks)
        
        prompt = f"""基于以下信息生成一个详细的综述大纲:
        查询主题: {query}
        综述类型: {review_type.value}
        上下文: {context}
        
        要求:
        1. 大纲层次分明,逻辑性强
        2. 符合学术综述的标准结构
        3. 确保覆盖主题的各个重要方面
        4. 以JSON格式返回,包含一级章节标题
        """
        
        # 4. 以JSON格式返回,包含章节标题和子标题

        response = self.llm.generate(prompt, verbose=verbose)
        json_outline = json.loads(response)
        sections = json_outline["章节"]
        return sections

    def generate_default_outline(self, review_type: ReviewType) -> List[str]:
        """规划综述结构"""
        structures = {
            ReviewType.CONCEPT: [
                {"title": "引言", "order": 1},
                {"title": "概念定义", "order": 2},
                {"title": "核心特征", "order": 3},
                {"title": "应用场景", "order": 4},
                {"title": "挑战与展望", "order": 5}
            ],
            ReviewType.STATUS: [
                {"title": "引言", "order": 1},
                {"title": "研究现状", "order": 2},
                {"title": "关键技术", "order": 3},
                {"title": "主要挑战", "order": 4},
                {"title": "未来趋势", "order": 5}
            ],
            ReviewType.COMPARISON: [
                {"title": "引言", "order": 1},
                {"title": "研究背景", "order": 2},
                {"title": "方法对比", "order": 3},
                {"title": "性能评估", "order": 4},
                {"title": "总结与建议", "order": 5}
            ],
            ReviewType.TIMELINE: [
                {"title": "引言", "order": 1},
                {"title": "技术起源", "order": 2},
                {"title": "发展历程", "order": 3},
                {"title": "关键突破", "order": 4},
                {"title": "未来展望", "order": 5}
            ]
        }
        return structures.get(review_type, [])
    
    def _prepare_context(self, query: str, review_type: ReviewType,
                        chunks: List[PaperChunk]) -> str:
        """准备用于生成大纲的上下文信息
        
        1. 对chunks进行信息提取和总结
        2. 根据综述类型组织信息
        """
        # 1. 提取每个chunk的关键信息
        summaries = []
        for chunk in chunks:
            prompt = f"""请从以下文献片段中提取关键信息:
            标题: {chunk.title}
            内容: {chunk.content}
            
            要求:
            1. 提取主要发现或方法
            2. 提取关键概念定义
            3. 提取研究背景或动机
            4. 以简洁的要点形式返回
            """
            summary = self.llm.generate(prompt, verbose=verbose)
            summaries.append(summary)
        
        # 2. 根据综述类型组织信息
        if review_type == ReviewType.CONCEPT:
            template = f"""基于以下文献内容,总结关于概念"{query}"的:
            1. 基本定义和内涵
            2. 主要特征或性质
            3. 应用场景
            4. 研究进展
            
            文献总结:
            {summaries}
            """
        
        elif review_type == ReviewType.STATUS:
            template = f"""基于以下文献内容,总结研究方向"{query}"的:
            1. 研究背景和意义
            2. 主要研究问题
            3. 目前的研究进展
            4. 存在的挑战
            5. 未来的发展趋势
            
            文献总结:
            {summaries}
            """
        
        elif review_type == ReviewType.COMPARISON:
            template = f"""基于以下文献内容,对比分析相关方法:
            1. 主要方法类别
            2. 各个方法的核心思想
            3. 优势和局限性
            4. 适用场景
            
            文献总结:
            {summaries}
            """
        
        else:  # ReviewType.EVOLUTION
            template = f"""基于以下文献内容,总结技术发展脉络:
            1. 发展阶段划分
            2. 各阶段的特征
            3. 关键技术突破
            4. 未来趋势
            
            文献总结:
            {summaries}
            """
            
        context = self.llm.generate(template, verbose=verbose)
        return context

def remove_escape_sequences(string):
    return string.encode('utf-8').decode('unicode_escape')

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

import numpy
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class ContentGenerator:
    """内容生成智能体,负责生成每个章节的具体内容"""
    
    def __init__(self, llm_client, max_tokens=2000):
        self.llm = llm_client
        self.max_tokens = max_tokens
        
    def generate_section(self, section_title: str, 
                        relevant_chunks: List[PaperChunk],
                        detailed_requirements: bool = False) -> Tuple[str, List[str]]:
        """生成单个章节的内容和引用"""
        
        # 1. 整理并分析输入内容
        # FIXME
        chunks_with_analysis = self._analyze_chunks(relevant_chunks)
        # chunks_with_analysis_file = "chunks_with_analysis.pkl"
        # if os.path.exists(chunks_with_analysis_file):
        #     chunks_with_analysis = pickle.load(open(chunks_with_analysis_file, "rb"))
        # else:
        #     chunks_with_analysis = self._analyze_chunks(relevant_chunks)
        #     pickle.dump(chunks_with_analysis, open(chunks_with_analysis_file, "wb"))

        
        # 2. 将chunks分批以适应上下文窗口
        chunks_batches = self._split_chunks(chunks_with_analysis)
        
        # 3. 分批生成内容
        section_content = []
        all_citations = set()
        
        for batch_idx, batch in enumerate(chunks_batches):
            # 准备生成提示词
            if detailed_requirements:
                prompt = self._get_detailed_prompt(section_title, batch, batch_idx, len(chunks_batches))
            else:
                prompt = self._get_basic_prompt(section_title, batch, batch_idx, len(chunks_batches))
            
            # 生成内容
            result = self.llm.generate(prompt, verbose=verbose)
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {result}")
                continue
            
            section_content.append(parsed_result["content"])
            all_citations.update(parsed_result["citations"])
            
            # 确保内容的连贯性
            if batch_idx > 0 and len(section_content) >= 2:
                transition_prompt = f"""请生成一个自然的过渡段落,连接以下两段内容:

                前一段结尾:
                {section_content[-2][-200:]}

                后一段开头:
                {section_content[-1][:200]}
                """
                transition = self.llm.generate(transition_prompt, verbose=verbose)
                section_content[-2] = section_content[-2] + "\n" + transition
        
        # 4. 合并并优化内容
        final_content = self._merge_and_polish(section_content, section_title)
        
        return final_content, list(all_citations)
    
    def _analyze_chunks(self, chunks: List[PaperChunk]) -> List[Dict]:
        """分析每个chunk的关键信息"""
        analyzed_chunks = []
        
        for chunk in chunks:
            prompt = f"""分析以下文献片段的关键信息:
            标题: {chunk.title}
            内容: {chunk.content}
            """ 
            prompt += dedent("""
            要求返回JSON格式:
            {
                "key_points": ["关键点1", "关键点2"...],
                "methodology": "使用的方法",
                "findings": "主要发现",
                "significance": "重要性说明"
            }
            """)
            
            generated_text = self.llm.generate(prompt, verbose=verbose)
            # generated_text = remove_escape_sequences(generated_text)
            analysis = json.loads(generated_text, cls=None)
            analyzed_chunks.append({
                "chunk": chunk,
                "analysis": analysis
            })
            
        return analyzed_chunks

    def _split_chunks(self, chunks: List[PaperChunk]) -> List[List[PaperChunk]]:
        """将chunks分批以适应LLM上下文窗口"""
        current_batch = []
        current_size = 0
        batches = []
        
        for chunk in chunks:
            chunk_size = len(chunk['chunk'].content)
            if current_size + chunk_size > self.max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            
            current_batch.append(chunk)
            current_size += chunk_size
            
        if current_batch:
            batches.append(current_batch)
            
        return batches
    
    def _format_chunks(self, chunks: List[PaperChunk]) -> str:
        """格式化chunks为输入文本"""
        formatted = []
        for chunk in chunks:
            formatted.append(f"Paper: {chunk.title} (ID: {chunk.paper_id})\n"
                           f"Content: {chunk.content}\n")
        return "\n".join(formatted)
    
    def _get_basic_prompt(self, section_title: str, batch: List[Dict], 
                         batch_idx: int, total_batches: int) -> str:
        """生成基本的内容生成提示词"""
        
        # 提取当前批次的关键信息
        key_points = []
        for item in batch:
            key_points.extend(item["analysis"]["key_points"])
        
        section_position = "开头" if batch_idx == 0 else "结尾" if batch_idx == total_batches - 1 else "中间"
        
        return f"""基于以下参考文献信息,为综述章节'{section_title}'生成{section_position}部分的内容:

        参考要点:
        {json.dumps(key_points, ensure_ascii=False, indent=2)}

        要求:
        1. 内容要有逻辑性和连贯性
        2. 准确引用参考文献
        3. 深入分析不同文献的观点
        4. 篇幅约800-1000字
        
        以JSON格式返回:
        {{
            "content": "生成的内容",
            "citations": ["引用的paper_id列表"]
        }}
        """
    
    def _get_detailed_prompt(self, section_title: str, batch: List[Dict],
                            batch_idx: int, total_batches: int) -> str:
        """生成详细的内容生成提示词"""
        # 整理当前批次的详细信息
        batch_info = {
            "key_points": [],
            "methodologies": set(),
            "findings": [],
            "significance": []
        }
        
        for item in batch:
            analysis = item["analysis"]
            batch_info["key_points"].extend(analysis["key_points"])
            batch_info["methodologies"].add(analysis["methodology"])
            batch_info["findings"].append(analysis["findings"])
            batch_info["significance"].append(analysis["significance"])
        
        section_position = "开头" if batch_idx == 0 else "结尾" if batch_idx == total_batches - 1 else "中间"
        
        return f"""作为一个专业的学术综述作者,请基于以下详细的参考文献信息,为综述章节'{section_title}'生成{section_position}部分的内容:

        核心要点:
        {json.dumps(batch_info["key_points"], ensure_ascii=False, indent=2)}

        研究方法:
        {json.dumps(list(batch_info["methodologies"]), ensure_ascii=False, indent=2)}

        主要发现:
        {json.dumps(batch_info["findings"], ensure_ascii=False, indent=2)}

        研究意义:
        {json.dumps(batch_info["significance"], ensure_ascii=False, indent=2)}

        详细要求:
        1. 内容结构要清晰,善用段落划分
        2. 准确理解和表达每篇文献的观点
        3. 对不同文献的观点进行对比和分析
        4. 指出研究中的创新点和局限性
        5. 揭示研究脉络和发展趋势
        6. 篇幅约1000-1200字
        
        以JSON格式返回:
        {{
            "content": "生成的内容",
            "citations": ["引用的paper_id列表"]
        }}
        """
    
    def _merge_and_polish(self, section_parts: List[str], section_title: str) -> str:
        """合并并优化章节内容"""
        merged_content = "\n".join(section_parts)
        
        polish_prompt = f"""请对以下综述内容进行优化和润色:
        章节标题: {section_title}
        
        原始内容:
        {merged_content}
        
        优化要求:
        1. 改善段落之间的过渡
        2. 确保专业术语使用一致
        3. 优化语言表达
        4. 增强论述的逻辑性
        5. 保持学术写作风格
        """
        
        polished_content = self.llm.generate(polish_prompt, verbose=verbose)
        return polished_content

class SegmentSummarizerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def summarize_in_batches(self, query: str, documents: List[Dict], review_type, batch_size=10) -> List[str]:
        partial_summaries = []
        all_citations = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i+batch_size]
            prompt, batch_citations = self._build_batch_prompt(query, batch_docs, review_type)
            all_citations.extend(batch_citations)
            summary_text = self.llm.generate(prompt, verbose=verbose)
            partial_summaries.append(summary_text)
        return partial_summaries, all_citations

    def _build_batch_prompt(self, query, batch_docs: List[Dict], review_type) -> str:
        references_text = []
        # for doc in batch_docs:
        #     snippet = doc["content"][:300]
        #     ref_str = f"{snippet}...[{doc['paper_id']}-{doc['chunk_id']}]"
        #     references_text.append(ref_str)
        batch_citations = []
        for paper_chunk in batch_docs:
            snippet = paper_chunk.content[:300]
            citation_id = f"{paper_chunk.paper_id}-{paper_chunk.chunk_id}"
            batch_citations.append(citation_id)
            ref_str = f"{snippet}...[{citation_id}]"
            references_text.append(ref_str)
        references_joined = "\n".join(references_text)

#         prompt = f"""
# 下面是一批论文片段，请你总结它们的关键信息并保留引用标注（[paper_id-chunk_id]），不要超过1500字。
# {references_joined}
# """
        prompt = f"""
你是一位学术研究助手。以下是一批文献内容片段，请为它们生成一个5000字以上的整合性的摘要。
摘要内容要保证全面、连贯，尽可能引用文献片段原文，避免使用不存在于文献内容片段的观点。
只参考有助于回答“{query}”{review_type_descriptions[review_type]}问题的片断，与问题无关的内容可以忽略。
不需要摘要标题和章节，以列表形式给出符合给出的文献片段观点的描述。
需要在对应信息后面标注引用，使用文献片段结尾的[paper_id-chunk_id]信息：

{references_joined}

请给出摘要：
"""
        return prompt, batch_citations

########################
# 4. 多层聚合
########################
class MultiLevelAggregatorAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def iterative_aggregate(self, query: str, partial_summaries: List[str], review_type: ReviewType, max_chunk_size=3) -> str:
        current_level = partial_summaries
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), max_chunk_size):
                chunk_summaries = current_level[i : i+max_chunk_size]
                input_text = "\n---\n".join(chunk_summaries)
                prompt = self._build_aggregation_prompt(query, input_text, review_type)
                aggregated = self.llm.generate(prompt, verbose=verbose)
                next_level.append(aggregated)
            current_level = next_level
        # 最终只剩一个大摘要
        return current_level[0]

#     def _build_aggregation_prompt(self, input_text: str, review_type: ReviewType) -> str:
#         prompt = f"""
# 请将以下多段摘要进行合并，写成一个更全面的总结性文本，并保留/合并引用标注：
# {input_text}
# """
        return prompt
    def _build_aggregation_prompt(self, query: str, input_text: str, review_type: ReviewType) -> str:
        """
        聚合多个 partial_summary 的 Prompt，需保留引用标签。
        """
        prompt = f"""
请将以下多段摘要进行合并，写成一个更加全面、连贯的5000字以上的总结性文本，保持关键信息不丢失，并保留各段的引用标注（[paper_id-chunk_id]）。
重点保留带有引用标注的文本片段的观点，可以直接引用，也可以根据需要进行合并。
如果有同一观点来自多个来源，可将引用合并在一起。
不需要摘要标题和章节，以列表形式给出符合给出的文献片段观点的描述。
只参考有助于回答“{query}”{review_type_descriptions[review_type]}问题的片断，与问题无关的内容可以忽略。

{input_text}

请输出合并后的文本：
"""
        return prompt

class StructuredWriterAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def write_review(self, query: str, final_summary: str, review_type: str) -> str:

# - 尽量保留带引用标注的文本片段的原文内容观点，并保留原有的引用标注 [paper_id-chunk_id]，可以直接引用，也可以根据需要进行合并和润色扩写。

        """
        根据不同的review_type对final_summary进行扩写，形成完整结构。
        """
        common_requirements = """
- 每一章节内容要有逻辑性和连贯性，全文应全面反映提牮的文献片段的观点。
- 尽量保留带引用标注的文本片段的原文内容观点，并将原来的引用格式 [paper_id-chunk_id] 改写成类似 "<sup>1</sup>" 的格式.
- At the end of the summary, list all references in the "References" section in the format "[1]: paper_id-chunk_id"
- 引用标注的内容观点至少20个以上。
- 综述全文不少于5000字:
        """

# 不少于5000字，并保留原有的引用标注 [paper_id-chunk_id]:

        # if review_type == "concept":
        if review_type == ReviewType.CONCEPT:
#             prompt = f"""
# 你是一位学术写作者。根据下列综合摘要，写一篇关于“{query}”{review_type_descriptions[review_type]}提问中技术概念的调研综述，包括：
# 1) 概念定义
# 2) 分类或子领域
# 3) 应用场景
# 4) 主要挑战
# 5) 未来展望
# 6) 参考文献

# {common_requirements}

# {final_summary}
# """

#             prompt = f"""
# 你是一位学术写作者。根据下列综合摘要，写一篇关于“{query}”{review_type_descriptions[review_type]}提问中技术概念的调研综述。

# {common_requirements}

# {final_summary}
# """

# 根据{review_type_descriptions[review_type]}类综述的通常结构（合适的综述标题，完整的综述大纲，主要内容控制在5-8个章节，不需要参考文献章节），并结合文献片段的观点，撰写一篇完整的{review_type_descriptions[review_type]}类综述文章。
# 合适的综述标题，主要章节包括：概念定义、分类或子领域、应用场景、主要挑战、未来展望，可根据提供的文献片段内容进行适当扩展,主要内容控制在6-8个章节，不需要参考文献章节。

            prompt = f"""
你是一位学术写作者。根据下列综合摘要，结合文献片段的观点，写一篇关于“{query}”{review_type_descriptions[review_type]}类综述文章。

- 合适的综述标题，主要章节包括：概念定义、分类或子领域、应用场景、主要挑战、未来展望等，可根据提供的文献片段内容进行适当扩展,主要内容控制在8-10个章节，注意调整章节的合理排列顺序，不需要参考文献章节。
- 将提供的文献片段内容分类归并到合适章节下,润色扩写成逻辑清晰、观点明明确、内容丰富的文字描述。避免只有一句话的章节，尽量保持章节内容的连贯性。
{common_requirements}

{final_summary}
"""
        # elif review_type == "direction":
        # elif review_type == "status":
        elif review_type == ReviewType.STATUS:
#             prompt = f"""
# 你是一位学术写作者。根据下列综合摘要，写一篇关于“{query}”{review_type_descriptions[review_type]}提问中研究方向的综述，包括:
# 1. 研究背景和意义
# 2. 主要研究问题
# 3. 目前的研究进展
# 4. 存在的挑战
# 5. 未来的发展趋势
# 6. 参考文献

# {common_requirements}

# {final_summary}
# """

            prompt = f"""
你是一位学术写作者。根据下列综合摘要，结合文献片段的观点，写一篇关于“{query}”{review_type_descriptions[review_type]}类综述文章。

- 合适的综述标题，主要章节包括：研究背景和意义、主要研究问题、目前的研究进展、存在的挑战、未来的发展趋势等，可根据提供的文献片段内容进行适当扩展,主要内容控制在8-10个章节，注意调整章节的合理排列顺序，不需要参考文献章节。
- 将提供的文献片段内容分类归并到合适章节下,润色扩写成逻辑清晰、观点明明确、内容丰富的文字描述。避免只有一句话的章节，尽量保持章节内容的连贯性。
{common_requirements}

{final_summary}
"""
        # elif review_type == "comparison":
        elif review_type == ReviewType.COMPARISON:
#             prompt = f"""
# 你是一位学术写作者。根据下列综合摘要，写一篇关于“{query}”{review_type_descriptions[review_type]}提问中对多种方法进行对比分析的综述，包括
# 1. 主要方法类别
# 2. 各个方法的核心思想
# 3. 优势和局限性
# 4. 适用场景
# 5. 参考文献

# {common_requirements}

# {final_summary}
# """
            prompt = f"""
你是一位学术写作者。根据下列综合摘要，结合文献片段的观点，写一篇关于“{query}”{review_type_descriptions[review_type]}类综述文章。

- 合适的综述标题，主要章节包括：主要方法类别、各个方法的核心思想、优势和局限性、适用场景等，可根据提供的文献片段内容进行适当扩展,主要内容控制在8-10个章节，注意调整章节的合理排列顺序，不需要参考文献章节。
- 将提供的文献片段内容分类归并到合适章节下,润色扩写成逻辑清晰、观点明明确、内容丰富的文字描述。避免只有一句话的章节，尽量保持章节内容的连贯性。
{common_requirements}

{final_summary}
"""
        # elif review_type == "evolution":
        # elif review_type == "timeline":
        elif review_type == ReviewType.TIMELINE:
#             prompt = f"""
# 你是一位学术写作者。根据下列综合摘要，写一篇关于“{query}”{review_type_descriptions[review_type]}提问中技术方法的发展脉络综述，包括
# 1. 发展阶段划分
# 2. 各阶段的特征
# 3. 关键技术突破
# 4. 未来趋势
# 5. 参考文献

# {common_requirements}

# {final_summary}
# """
            prompt = f"""
你是一位学术写作者。根据下列综合摘要，结合文献片段的观点，写一篇关于“{query}”{review_type_descriptions[review_type]}类综述文章。

- 合适的综述标题，主要章节包括：发展阶段划分、各阶段的特征、关键技术突破、未来趋势等，可根据提供的文献片段内容进行适当扩展,主要内容控制在8-10个章节，注意调整章节的合理排列顺序，不需要参考文献章节。
- 将提供的文献片段内容分类归并到合适章节下,润色扩写成逻辑清晰、观点明明确、内容丰富的文字描述。避免只有一句话的章节，尽量保持章节内容的连贯性。
{common_requirements}

{final_summary}
"""
        else:
#             prompt = f"""
# 你是一位学术写作者。根据下列综合摘要，写一篇文献综述, 包括:
# 引言
# 主体
# 相关工作
# 结论
# 参考文献

# {common_requirements}

# {final_summary}
# """
            prompt = f"""
你是一位学术写作者。根据下列综合摘要，结合文献片段的观点，写一篇关于“{query}”{review_type_descriptions[review_type]}类综述文章。

- 合适的综述标题，主要章节包括：引言、主体、相关工作、结论等，可根据提供的文献片段内容进行适当扩展,主要内容控制在8-10个章节，注意调整章节的合理排列顺序，不需要参考文献章节。
- 将提供的文献片段内容分类归并到合适章节下,润色扩写成逻辑清晰、观点明明确、内容丰富的文字描述。避免只有一句话的章节，尽量保持章节内容的连贯性。
{common_requirements}

{final_summary}
"""

        full_text = self.llm.generate(prompt, verbose=verbose)
        return full_text

#     def _build_aggregation_prompt(self, input_text: str) -> str:
#         """
#         聚合多个 partial_summary 的 Prompt，需保留引用标签。
#         """
#         prompt = f"""
# 请将以下多段摘要进行合并，写成一个更加精炼、连贯的总结性文本，保持关键信息不丢失，并保留各段的引用标注（[paper_id-chunk_id]）。
# 如果有同一观点来自多个来源，可将引用合并在一起。

# {input_text}

# 请输出合并后的文本：
# """
#         return prompt

########################
# 6. 引用校验 & 质量审校
########################
class CitationVerifierAgent:
    def verify_citations(self, text: str, original_docs: List[Dict]) -> str:
        """
        简化：暂不做深入校验，只做一个长度检测 & 占位。
        可扩展为: 逐条解析 [paper_id-chunk_id], 回溯原文做相似度检查等.
        """
        """
        1. 在最终文本中查找所有出现的 [paper_id-chunk_id]。
        2. 对应到 original_docs，做一个简单的事实一致性核验（可用问答或向量相似度等策略）。
        3. 对不匹配的引用做标记或删除，并提示模型二次修订。
        """
        return text  # 暂不修改

class QualityAgent:
    def review_and_refine(self, text: str) -> str:
        # if len(text) < 2000:
        #     # 若不够长, 进行简单补充
        #     text += "\n\n(自动补充内容以满足2000字要求) " + "延伸讨论..."*100
        return text


class ReviewOrchestrator:
    """系统协调器,负责整体流程控制"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.query_expander = QueryExpander(llm_client)
        self.retriever = ContentRetriever(llm_client)
        self.outline_generator = OutlineGenerator(llm_client)
        self.content_generator = ContentGenerator(llm_client)

        self.segment_summarizer = SegmentSummarizerAgent(llm_client)
        self.aggregator = MultiLevelAggregatorAgent(llm_client)
        self.writer = StructuredWriterAgent(llm_client)
        self.citation_verifier = CitationVerifierAgent()
        self.quality_agent = QualityAgent()
        
    # def evaluate_quality(self, review_content: Dict, original_chunks: List[PaperChunk]) -> ReviewQuality:
    #     """评估综述质量"""
    #     quality = ReviewQuality()
        
    #     # 1. 评估文献覆盖度
    #     cited_papers = set()
    #     for section in review_content["content"]:
    #         for citation in section["citations"]:
    #             cited_papers.add(citation)
    #     coverage = len(cited_papers) / len({chunk.paper_id for chunk in original_chunks})
    #     quality.update('coverage', coverage)
        
    #     # 2. 评估结构完整性
    #     prompt = f"""评估以下综述的结构完整性(0-1分):
    #     类型: {review_content["review_type"]}
    #     大纲: {[section["title"] for section in review_content["content"]]}
        
    #     评分标准:
    #     1. 结构是否完整(引言、主体、总结等)
    #     2. 章节层次是否合理
    #     3. 逻辑关系是否清晰
    #     4. 是否符合该类型综述的标准结构
        
    #     只返回分数数字
    #     """
    #     structure_score = float(self.llm.generate(prompt, verbose=verbose))
    #     quality.update('structure', structure_score)
        
    #     # 3. 评估内容连贯性
    #     all_content = "\n".join(section["content"] for section in review_content["content"])
    #     prompt = f"""评估以下综述的内容连贯性(0-1分):
    #     {all_content[:2000]}  # 评估前2000字
        
    #     评分标准:
    #     1. 段落之间是否衔接自然
    #     2. 论述是否层次分明
    #     3. 是否存在重复或矛盾
    #     4. 专业术语使用是否准确一致
        
    #     只返回分数数字
    #     """
    #     coherence_score = float(self.llm.generate(prompt))
    #     quality.update('coherence', coherence_score)
        
    #     # 4. 评估引用准确性
    #     # 随机抽查5个引用
    #     import random
    #     sample_citations = random.sample(list(cited_papers), min(5, len(cited_papers)))
    #     citation_scores = []
        
    #     for paper_id in sample_citations:
    #         # 获取原文内容
    #         paper_chunks = [c for c in original_chunks if c.paper_id == paper_id]
    #         original_content = "\n".join(c.content for c in paper_chunks)
            
    #         # 找到综述中引用该文献的段落
    #         citing_passages = []
    #         for section in review_content["content"]:
    #             if paper_id in section["citations"]:
    #                 citing_passages.append(section["content"])
            
    #         prompt = f"""评估引用准确性(0-1分):
    #         原文: {original_content[:1000]}
    #         综述中的引用: {' '.join(citing_passages)[:500]}
            
    #         评分标准:
    #         1. 是否准确理解原文观点
    #         2. 是否存在断章取义
    #         3. 引用位置是否恰当
    #         4. 是否有效支持论述
            
    #         只返回分数数字
    #         """
    #         score = float(self.llm.generate(prompt, verbose=verbose))
    #         citation_scores.append(score)
        
    #     quality.update('citation', sum(citation_scores) / len(citation_scores))
        
    #     # 5. 评估见解新颖性
    #     prompt = f"""评估以下综述的见解新颖性(0-1分):
    #     综述内容: {all_content[:2000]}
        
    #     评分标准:
    #     1. 是否提出新的观点或视角
    #     2. 是否有效总结研究趋势
    #     3. 是否深入分析问题本质
    #     4. 是否提供有价值的未来展望
        
    #     只返回分数数字
    #     """
    #     novelty_score = float(self.llm.generate(prompt, verbose=verbose))
    #     quality.update('novelty', novelty_score)
        
    #     return quality

    def do_expand_query(self, query: str, review_type: ReviewType) -> List[str]:
        """扩展查询"""
        return self.query_expander.expand_query(query, review_type)

    def retrieve_chunks(self, queries: List[str], min_papers=50) -> List[PaperChunk]:
        chunks = self.retriever.retrieve_by_queries(queries)
        logger.info(f"Retrieved {len(chunks)} chunks")
        if len(chunks) < min_papers:
            print(f"Warning: Only found {len(chunks)} relevant chunks, less than minimum requirement of {min_papers}")
        
        return chunks


    def generate_review(self, query: str, review_type: ReviewType) -> Dict:

        root_dir = f"outputs/{query}"
        os.makedirs(root_dir, exist_ok=True)

        """生成完整的文献综述"""
        # 1. 扩展查询
        @cache_or_rebuild(cache_file=f"{root_dir}/expanded_queries.json")
        def do_expand_query(query: str, review_type: ReviewType) -> List[str]:
            """扩展查询"""
            return self.query_expander.expand_query(query, review_type)
        expanded_queries = do_expand_query(query, review_type)
        
        # 2. 检索内容
        @cache_or_rebuild(cache_file=f"{root_dir}/chunks.pkl")
        def do_retrieve_chunks(expanded_queries: List[str], min_papers=50) -> List[PaperChunk]:
            return self.retrieve_chunks(expanded_queries, min_papers=min_papers)
        chunks = do_retrieve_chunks(expanded_queries, min_papers=50)

        @cache_or_rebuild(cache_file=f"{root_dir}/clustered_chunks.pkl")
        def do_cluster_chunks(chunks: List[PaperChunk]) -> Dict:
            return self.retriever.cluster_chunks(chunks)
        clustered_chunks = do_cluster_chunks(chunks)

        # 3) 分批摘要
        @cache_or_rebuild(cache_file=f"{root_dir}/partial_summaries.pkl")
        def do_partial_summaries(query: str, chunks: List[PaperChunk], review_type, batch_size=5) -> Tuple[List[str], List[str]]:
            return self.segment_summarizer.summarize_in_batches(query, chunks, review_type=review_type, batch_size=batch_size)
        partial_summaries, partial_citations = do_partial_summaries(query, chunks, review_type=review_type, batch_size=5)

        partial_citations_file = f"{root_dir}/partial_citations.txt"
        with open(partial_citations_file, "w", encoding="utf-8") as f:
            f.write("\n".join(partial_citations))

        # 4) 多层聚合
        @cache_or_rebuild(cache_file=f"{root_dir}/global_summary.pkl")
        def do_global_summary(query: str, partial_summaries: List[str], review_type: ReviewType, max_chunk_size=2) -> str:
            return self.aggregator.iterative_aggregate(query, partial_summaries, review_type=review_type, max_chunk_size=max_chunk_size)
        global_summary = do_global_summary(query, partial_summaries, review_type=review_type, max_chunk_size=2)

        # 5) 结构化写作
        @cache_or_rebuild(cache_file=f"{root_dir}/structured_review.pkl")
        def do_write_review(query: str, global_summary: str, review_type: ReviewType) -> Dict:
            return self.writer.write_review(query, global_summary, review_type)
        structured_review = do_write_review(query, global_summary, review_type)

        # 6) 引用校验
        @cache_or_rebuild(cache_file=f"{root_dir}/verified_text.pkl")
        def do_verify_citations(structured_review: Dict, chunks: List[PaperChunk]) -> Dict:
            return self.citation_verifier.verify_citations(structured_review, chunks)
        verified_text = do_verify_citations(structured_review, chunks)

        # 7) 质量审校
        @cache_or_rebuild(cache_file=f"{root_dir}/review_content.pkl")
        def do_quality_review(verified_text: str) -> str:
            return self.quality_agent.review_and_refine(verified_text)
        review_content = do_quality_review(verified_text)

        all_references_file = f"{root_dir}/all_references.jsonl"
        if os.path.exists(all_references_file):
            references = [json.loads(line.strip()) for line in open(all_references_file, "r", encoding="utf-8")]
        else:
            references = generate_references(citations_file=partial_citations_file, kb=kb)
            with open(all_references_file, "w", encoding="utf-8") as fd:
                for ref in references:
                    line = json.dumps(ref, ensure_ascii=False)
                    fd.write(f"{line}\n")


        selected_references = []
        references_text = "\n## 参考文献\n\n"
        for ref in references:
           paper_id = ref["paper_id"] 
           chunk_id = ref["chunk_id"]
           citation_id = f"{paper_id}-{chunk_id}"
           if citation_id in review_content:
               ref_line = ref['ref']
               references_text += f"{ref_line}\n"
               selected_references.append(ref)

               review_content = review_content.replace(f"{paper_id}-{chunk_id}", f"{ref_line}, chunk {chunk_id}")
        # review_content += references_text

        selected_references_file = f"{root_dir}/selected_references.jsonl"
        if os.path.exists(selected_references_file):
            selected_references = [json.loads(line.strip()) for line in open(selected_references_file, "r", encoding="utf-8")]
        else:
            with open(selected_references_file, "w", encoding="utf-8") as fd:
                for ref in selected_references:
                    line = json.dumps(ref, ensure_ascii=False)
                    fd.write(f"{line}\n")

        output_file = f"{root_dir}/review_{query}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(review_content)
        logger.info(f"综述已保存到文件: {output_file}")

        result = {
            "query": query,
            "review_type": review_type.value,
            "content": review_content,
            "citations": [], #list(set(all_citations)),  # 去重
            "total_words": 0, #total_words,
            "metrics": {}
        }

        return result
        

    def generate_review_sonnet(self, query: str, review_type: ReviewType) -> Dict:
        """生成完整的文献综述"""
        # 1. 扩展查询
        # expanded_queries = self.query_expander.expand_query(query, review_type)
        expaned_queries_file = "expanded_queries.json"
        if os.path.exists(expaned_queries_file):
            expanded_queries = json.load(open(expaned_queries_file, "r", encoding="utf-8"))
        else:
            expanded_queries = self.do_expand_query(query, review_type)
            with open(expaned_queries_file, "w", encoding="utf-8") as f:
                json.dump(expanded_queries, f, ensure_ascii=False, indent=2)
        
        # 2. 检索内容

        chunks_file = "chunks.pkl"
        if os.path.exists(chunks_file):
            # chunks = json.load(open(chunks_file, "r", encoding="utf-8"))
            chunks = pickle.load(open(chunks_file, "rb"))
        else:
            chunks = self.do_retrieve_chunks(expanded_queries, min_papers=50)
            # with open(chunks_file, "w", encoding="utf-8") as f:
            #     json.dump(chunks, f, ensure_ascii=False, indent=2)
            pickle.dump(chunks, open(chunks_file, "wb"))

        clustered_chunks_file = "clustered_chunks.pkl"
        if os.path.exists(clustered_chunks_file):
            # clustered_chunks = json.load(open(clustered_chunks_file, "r", encoding="utf-8"))
            clustered_chunks = pickle.load(open(clustered_chunks_file, "rb"))
        else:
            clustered_chunks = self.retriever.cluster_chunks(chunks)
            # with open(clustered_chunks_file, "w", encoding="utf-8") as f:
            #     json.dump(clustered_chunks, f, ensure_ascii=False, indent=2)
            pickle.dump(clustered_chunks, open(clustered_chunks_file, "wb"))
        
        # 3. 生成大纲
        # outline = self.outline_generator.generate_outline(
        #     query, review_type, chunks[:10]  # 使用部分chunks生成大纲
        # )
        outline = self.outline_generator.generate_default_outline(review_type)
        
        # 4. 按章节生成内容
        review_content = []
        all_citations = []
        total_words = 0
        
        for section in outline:
            # section_title = section["章节标题"]
            # section_title += " - " + "/".join(section["子标题"])
            section_title = section["title"]
            # FIXME
            section_chunks = self._get_relevant_chunks(
                section_title, 
                clustered_chunks
            )
            # section_chunks_file = f"{section_title}_chunks.pkl"
            # if os.path.exists(section_chunks_file):
            #     section_chunks = pickle.load(open(section_chunks_file, "rb"))
            # else:
            #     section_chunks = self._get_relevant_chunks(
            #         section_title, 
            #         clustered_chunks
            #     )
            #     pickle.dump(section_chunks, open(section_chunks_file, "wb"))
            
            # 确保每个章节至少有一定数量的相关文献
            if len(section_chunks) < 10:
                print(f"Warning: Section '{section['title']}' only has {len(section_chunks)} relevant chunks")
                # 补充检索
                additional_chunks = kb.search_papers(
                    f"{query} {section_title}", 
                    top_k=20
                )
                additional_chunks = [kb_chunk_to_paper_chunk(c) for c in additional_chunks]
                section_chunks.extend([c for c in additional_chunks if c.chunk_id not in {x.chunk_id for x in section_chunks}])
            
            content, citations = self.content_generator.generate_section(
                section_title,
                section_chunks
            )

            
            # 检查内容质量
            prompt = f"""评估以下内容是否需要优化(返回yes/no):
            {content[:1000]}
            
            评估标准:
            1. 内容是否充实
            2. 论述是否深入
            3. 观点是否明确
            4. 是否有效利用引用材料
            """
            need_optimization = self.llm.generate(prompt, verbose=verbose).strip().lower() == "yes"
            
            if need_optimization:
                # 使用更多上下文重新生成
                content, citations = self.content_generator.generate_section(
                    section_title,
                    section_chunks,
                    detailed_requirements=True
                )
            
            review_content.append({
                "title": section_title,
                "content": content,
                "citations": citations,
                "word_count": len(content.split())
            })
            
            all_citations.extend(citations)
            total_words += len(content.split())
            
            # 检查总字数是否达到要求
            if total_words < 2000 and section == outline[-1]:
                print("Warning: Review length below requirement, expanding content...")
                # 为每个章节补充内容
                for section_data in review_content:
                    additional_content, add_citations = self._expand_section_content(
                        section_data["title"],
                        section_data["content"],
                        clustered_chunks
                    )
                    section_data["content"] += "\n" + additional_content
                    section_data["citations"].extend(add_citations)
                    section_data["word_count"] = len(section_data["content"].split())
                    total_words += len(additional_content.split())
        
        result = {
            "query": query,
            "review_type": review_type.value,
            "content": review_content,
            "citations": list(set(all_citations)),  # 去重
            "total_words": total_words,
            "metrics": {}
        }
        
        # # 评估质量
        # quality = self.evaluate_quality(result, chunks)
        # result["metrics"] = quality.metrics
        # result["overall_score"] = quality.get_overall_score()
        
        return result
        
    def _expand_section_content(self, title: str, existing_content: str, 
                              clustered_chunks: Dict[str, List[PaperChunk]]) -> Tuple[str, List[str]]:
        """扩展章节内容"""
        prompt = f"""分析以下章节内容,指出需要补充的方面:
        标题: {title}
        现有内容: {existing_content}
        
        以逗号分隔的关键词形式返回需要补充的方面
        """
        aspects_to_expand = self.llm.generate(prompt, verbose=verbose).strip().split(",")
        
        additional_chunks = []
        for aspect in aspects_to_expand:
            # 检索相关内容
            relevant_chunks = kb.search_papers(
                f"{title} {aspect}",
                top_k=10
            )
            relevant_chunks = [kb_chunk_to_paper_chunk(c) for c in relevant_chunks]
            additional_chunks.extend(relevant_chunks)
        
        # 生成补充内容
        additional_content, citations = self.content_generator.generate_section(
            f"{title} - {' '.join(aspects_to_expand)}",
            additional_chunks
        )
        
        return additional_content, citations
    
    def _get_relevant_chunks(self, section_title: str, 
                           clustered_chunks: Dict[str, List[PaperChunk]]) -> List[PaperChunk]:
        """获取与特定章节相关的chunks
        
        1. 计算章节标题与各个簇的相关度
        2. 选择最相关的簇
        3. 对选中簇中的chunks进行相关度排序
        """
        # 1. 构建章节主题的特征描述
        prompt = f"""请详细描述章节"{section_title}"可能包含的主要内容和关键概念:
        1. 核心主题
        2. 相关概念
        3. 可能涉及的方法或技术
        4. 重要性和影响
        以逗号分隔的关键词列表形式返回
        """
        section_keywords = set(self.llm.generate(prompt, verbose=verbose).strip().split(","))
        
        # 2. 计算与各个簇的相关度
        cluster_relevance = {}
        for topic, chunks in clustered_chunks.items():
            # 获取簇的关键词特征
            cluster_text = f"{topic}\n" + "\n".join(
                f"{chunk.title}\n{chunk.content[:200]}" 
                for chunk in chunks[:5]  # 使用前5个chunk作为代表
            )
            prompt = f"""请从以下文本中提取关键概念,以逗号分隔:
            {cluster_text}
            """
            cluster_keywords = set(self.llm.generate(prompt, verbose=verbose).strip().split(","))
            
            # 计算关键词重叠度作为相关度分数
            relevance = len(section_keywords & cluster_keywords) / len(section_keywords | cluster_keywords)
            cluster_relevance[topic] = relevance
        
        # 3. 选择最相关的簇(相关度大于阈值)
        relevant_chunks = []
        for topic, relevance in sorted(cluster_relevance.items(), key=lambda x: x[1], reverse=True):
            if relevance > 0.2:  # 相关度阈值
                chunks = clustered_chunks[topic]
                # 对簇内chunks进行相关度排序
                chunk_scores = []
                for chunk in chunks:
                    prompt = f"""评估以下文献片段与主题"{section_title}"的相关度(0-1):
                    标题: {chunk.title}
                    内容: {chunk.content[:300]}
                    只返回分数数字
                    """
                    score = float(self.llm.generate(prompt, verbose=verbose))
                    chunk_scores.append((chunk, score))
                
                # 添加相关度较高的chunks
                sorted_chunks = [c for c, s in sorted(chunk_scores, key=lambda x: x[1], reverse=True)
                               if s > 0.3]  # 相关度阈值
                relevant_chunks.extend(sorted_chunks)
        
        return relevant_chunks

def do_summary(args):
    query = args.query

    model_name = os.getenv("OPENAI_DEFAULT_MODEL")
    llm_client = LLMClient(model_name=model_name)

    from intent_recognition import classify_intent_with_prompt
    review_type = classify_intent_with_prompt(llm_client, query)

    try:
        review_type = ReviewType(review_type)
    except ValueError:
        raise ValueError(f"Invalid review type: {args.review_type}")
    print(f"Query: {query}\nReview Type: {review_type_descriptions[review_type]}\n")

    orchestrator = ReviewOrchestrator(llm_client=llm_client)
    
    logger.info(f"\n处理查询: {query}")
    result = orchestrator.generate_review(query, review_type)
    logger.debug(f"{result=}")


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="文献综述生成器")
    parser.add_argument("--do_summary", action="store_true", help="是否执行综述生成")
    parser.add_argument("--query", type=str, default="损失函数", help="查询问题")
    parser.add_argument("--review_type", type=str, default="concept", help="综述类型")
    args = parser.parse_args()
    return args

async def main():
    args = get_args()

    # 示例查询
    # queries = [
    #     ("损失函数", ReviewType.CONCEPT),
    #     ("Text2SQL研究现状如何，面临哪些挑战？", ReviewType.STATUS),
    #     ("有哪些方法可以提升大模型的规划能力，各自优劣是什么？", ReviewType.COMPARISON)
    #     ("多模态大模型的技术发展路线是什么样的？", ReviewType.TIMELINE)
    # ]

    if args.do_summary:
        do_summary(args)
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())