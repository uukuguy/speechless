
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


# import nltk
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# # 下载必要的NLTK数据
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


class QueryExpander:
    """查询扩展智能体,负责将用户简短查询扩展为更全面的检索关键词"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def expand_query(self, query: str, review_type: ReviewType, lang="chinese") -> List[str]:
        """扩展用户查询为多个相关查询"""
        if lang == "chinese":
            prompt = f"""请基于以下用户查询,生成5-8个相关的检索关键词或短语,以帮助全面检索相关文献:
            用户查询: {query}
            综述类型: {review_type_descriptions[review_type]}
            要求:
            1. 关键词应该覆盖该主题的不同方面
            2. 包含该领域的专业术语
            3. 考虑近义词和相关概念
            4. 以JSON列表格式返回
            """
        elif lang == "english":
            prompt = f"""Please generate 5-8 relevant search keywords or phrases based on the following user query to help comprehensively retrieve related literature:  
            User Query: {query}  
            Review Type: {review_type_descriptions[review_type]}  
            Requirements:  
            1. Keywords should cover different aspects of the topic.  
            2. Include terminology specific to the field.  
            3. Consider synonyms and related concepts.  
            4. Return in JSON list format
            """
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
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
    
    def cluster_chunks(self, chunks: List[PaperChunk], lang="chinese") -> Dict[str, List[PaperChunk]]:
        """将检索到的内容按主题聚类
        
        使用LLM为每个chunk生成主题标签,然后根据标签相似度聚类
        """
        # 1. 为每个chunk生成主题标签
        chunk_topics = {}
        bad_chunks = []
        for chunk in chunks:
            if lang == "chinese":
                prompt = f"""请为以下文献片段生成3个主题标签,以逗号分隔:
                标题: {chunk.title}
                内容: {chunk.content[:1500]}  # 使用前500字符以节省token
                """
            elif lang == "english":
                prompt = f"""Please generate 3 topic tags for the following paper chunk, separated by commas:  
                Title: {chunk.title}  
                Content: {chunk.content[:1500]}  # Using the first 500 characters to save tokens
                """
            else:
                raise ValueError(f"Unsupported language: {lang}")

            response = self.llm.generate(prompt, verbose=verbose)
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
                        sample_chunks: List[PaperChunk], lang="chinese") -> List[str]:
        """生成综述大纲"""
        # 构建输入上下文
        context = self._prepare_context(query, review_type, sample_chunks)
        
        if lang == "chinese":
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
        elif lang == "english":
            prompt = f"""Generate a detailed outline based on the following information:
            Query Topic: {query}
            Review Type: {review_type.value}
            Context: {context}
            
            Requirements:
            1. Clear and logical outline structure
            2. Conform to the standard structure of academic reviews
            3. Ensure coverage of all important aspects of the topic
            4. Return in JSON format, including top-level section titles
            """
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
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
                        chunks: List[PaperChunk], lang="chinese") -> str:
        """准备用于生成大纲的上下文信息
        
        1. 对chunks进行信息提取和总结
        2. 根据综述类型组织信息
        """
        # 1. 提取每个chunk的关键信息
        summaries = []
        for chunk in chunks:
            if lang == "chinese":
                prompt = f"""请从以下文献片段中提取关键信息:
                标题: {chunk.title}
                内容: {chunk.content}
                
                要求:
                1. 提取主要发现或方法
                2. 提取关键概念定义
                3. 提取研究背景或动机
                4. 以简洁的要点形式返回
                """
            elif lang == "english":
                prompt = f"""Extract key information from the following paper chunk:
                Title: {chunk.title}
                Content: {chunk.content}
                
                Requirements:
                1. Extract main findings or methods
                2. Extract key concept definitions
                3. Extract research background or motivation
                4. Return in concise bullet points
                """
            else:
                raise ValueError(f"Unsupported language: {lang}")

            summary = self.llm.generate(prompt, verbose=verbose)
            summaries.append(summary)
        
        # 2. 根据综述类型组织信息
        if review_type == ReviewType.CONCEPT:
            if lang == "chinese":
                template = f"""基于以下文献内容,总结关于概念"{query}"的:
                1. 基本定义和内涵
                2. 主要特征或性质
                3. 应用场景
                4. 研究进展
                
                文献总结:
                {summaries}
                """
            elif lang == "english":
                template = f"""Based on the following literature, summarize the concept "{query}" with:
                1. Basic definition and connotation
                2. Main features or properties
                3. Application scenarios
                4. Research progress
                
                Literature Summaries:
                {summaries}
                """
            else:
                raise ValueError(f"Unsupported language: {lang}")
        
        elif review_type == ReviewType.STATUS:
            if lang == "chinese":
                template = f"""基于以下文献内容,总结研究方向"{query}"的:
                1. 研究背景和意义
                2. 主要研究问题
                3. 目前的研究进展
                4. 存在的挑战
                5. 未来的发展趋势
                
                文献总结:
                {summaries}
                """
            elif lang == "english":
                template = f"""Based on the following literature, summarize the research direction "{query}" with:
                1. Research background and significance
                2. Main research questions
                3. Current research progress
                4. Existing challenges
                5. Future development trends
                
                Literature Summaries:
                {summaries}
                """
            else:
                raise ValueError(f"Unsupported language: {lang}")
        
        elif review_type == ReviewType.COMPARISON:
            if lang == "chinese":
                template = f"""基于以下文献内容,对比分析相关方法:
                1. 主要方法类别
                2. 各个方法的核心思想
                3. 优势和局限性
                4. 适用场景
                
                文献总结:
                {summaries}
                """
            elif lang == "english":
                template = f"""Based on the following literature, compare and analyze relevant methods with:
                1. Main categories of methods
                2. Core ideas of each method
                3. Advantages and limitations
                4. Application scenarios
                
                Literature Summaries:
                {summaries}
                """
            else:
                raise ValueError(f"Unsupported language: {lang}")
        
        else:  # ReviewType.EVOLUTION
            if lang == "chinese":
                template = f"""基于以下文献内容,总结技术发展脉络:
                1. 发展阶段划分
                2. 各阶段的特征
                3. 关键技术突破
                4. 未来趋势
                
                文献总结:
                {summaries}
                """
            elif lang == "english":
                template = f"""Based on the following literature, summarize the technical evolution with:
                1. Division of development stages
                2. Characteristics of each stage
                3. Key technological breakthroughs
                4. Future trends
                
                Literature Summaries:
                {summaries}
                """
            else:
                raise ValueError(f"Unsupported language: {lang}")
            
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

class ContentGenerator:
    """内容生成智能体,负责生成每个章节的具体内容"""
    
    def __init__(self, llm_client, max_tokens=2000):
        self.llm = llm_client
        self.max_tokens = max_tokens
        
    def generate_section(self, section_title: str, 
                        relevant_chunks: List[PaperChunk],
                        detailed_requirements: bool = False, lang="chinese") -> Tuple[str, List[str]]:
        """生成单个章节的内容和引用"""
        
        # 1. 整理并分析输入内容
        # FIXME
        chunks_with_analysis = self._analyze_chunks(relevant_chunks, lang=lang)
        # chunks_with_analysis_file = "chunks_with_analysis.pkl"
        # if os.path.exists(chunks_with_analysis_file):
        #     chunks_with_analysis = pickle.load(open(chunks_with_analysis_file, "rb"))
        # else:
        #     chunks_with_analysis = self._analyze_chunks(relevant_chunks, lang=lang)
        #     pickle.dump(chunks_with_analysis, open(chunks_with_analysis_file, "wb"))

        
        # 2. 将chunks分批以适应上下文窗口
        chunks_batches = self._split_chunks(chunks_with_analysis)
        
        # 3. 分批生成内容
        section_content = []
        all_citations = set()
        
        for batch_idx, batch in enumerate(chunks_batches):
            # 准备生成提示词
            if detailed_requirements:
                prompt = self._get_detailed_prompt(section_title, batch, batch_idx, len(chunks_batches), lang=lang)
            else:
                prompt = self._get_basic_prompt(section_title, batch, batch_idx, len(chunks_batches), lang=lang)
            
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

                if lang == "chinese":
                    transition_prompt = f"""请生成一个自然的过渡段落,连接以下两段内容:

                    前一段结尾:
                    {section_content[-2][-200:]}

                    后一段开头:
                    {section_content[-1][:200]}
                    """
                elif lang == "english":
                    transition_prompt = f"""Generate a natural transition paragraph to connect the following two sections:

                    End of the previous section:
                    {section_content[-2][-200:]}

                    Beginning of the next section:
                    {section_content[-1][:200]}
                    """
                else:
                    raise ValueError(f"Unsupported language: {lang}")

                transition = self.llm.generate(transition_prompt, verbose=verbose)
                section_content[-2] = section_content[-2] + "\n" + transition
        
        # 4. 合并并优化内容
        final_content = self._merge_and_polish(section_content, section_title, lang=lang)
        
        return final_content, list(all_citations)
    
    def _analyze_chunks(self, chunks: List[PaperChunk], lang="chinese") -> List[Dict]:
        """分析每个chunk的关键信息"""
        analyzed_chunks = []
        
        for chunk in chunks:
            if lang == "chinese":
                prompt = f"""分析以下文献片段的关键信息:
                标题: {chunk.title}
                内容: {chunk.content}
                """ 
            elif lang == "english":
                prompt = f"""Analyze the key information of the following paper chunk:
                Title: {chunk.title}
                Content: {chunk.content}
                """
            else:
                raise ValueError(f"Unsupported language: {lang}")
            
            if lang == "chinese":
                prompt += dedent("""
                要求返回JSON格式:
                {
                    "key_points": ["关键点1", "关键点2"...],
                    "methodology": "使用的方法",
                    "findings": "主要发现",
                    "significance": "重要性说明"
                }
                """)
            elif lang == "english":
                prompt += dedent("""
                Return in JSON format:
                {
                    "key_points": ["Key point 1", "Key point 2"...],
                    "methodology": "Methodology used",
                    "findings": "Main findings",
                    "significance": "Significance description"
                }
                """)
            else:
                raise ValueError(f"Unsupported language: {lang}")
            
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
                         batch_idx: int, total_batches: int, lang="chinese") -> str:
        """生成基本的内容生成提示词"""
        
        # 提取当前批次的关键信息
        key_points = []
        for item in batch:
            key_points.extend(item["analysis"]["key_points"])
        
        if lang == "chinese":
            section_position = "开头" if batch_idx == 0 else "结尾" if batch_idx == total_batches - 1 else "中间"
        else:
            section_position = "beginning" if batch_idx == 0 else "end" if batch_idx == total_batches - 1 else "middle"
        
        if lang == "chinese":
            prompt = f"""基于以下参考文献信息,为综述章节'{section_title}'生成{section_position}部分的内容:

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
        elif lang == "english":
            prompt = f"""Based on the following reference information, generate the {section_position} part of the content for the review section '{section_title}':

            Key Points:
            {json.dumps(key_points, ensure_ascii=False, indent=2)}

            Requirements:
            1. Ensure logical and coherent content
            2. Accurately cite the reference literature
            3. In-depth analysis of viewpoints from different literature
            4. Length: approximately 800-1000 words
            
            Return in JSON format:
            {{
                "content": "Generated content",
                "citations": ["List of cited paper_ids"]
            }}
            """

        return prompt
    
    def _get_detailed_prompt(self, section_title: str, batch: List[Dict],
                            batch_idx: int, total_batches: int, lang="chinese") -> str:
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
        
        if lang == "chinese":
            section_position = "开头" if batch_idx == 0 else "结尾" if batch_idx == total_batches - 1 else "中间"
        else:
            section_position = "beginning" if batch_idx == 0 else "end" if batch_idx == total_batches - 1 else "middle"
        
        if lang == "chinese":
            prompt = f"""作为一个专业的学术综述作者,请基于以下详细的参考文献信息,为综述章节'{section_title}'生成{section_position}部分的内容:

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
        elif lang == "english":
            prompt = f"""As a professional academic reviewer, based on the following detailed reference information, generate the {section_position} part of the content for the review section '{section_title}':

            Key Points:
            {json.dumps(batch_info["key_points"], ensure_ascii=False, indent=2)}

            Methodologies:
            {json.dumps(list(batch_info["methodologies"]), ensure_ascii=False, indent=2)}

            Main Findings:
            {json.dumps(batch_info["findings"], ensure_ascii=False, indent=2)}

            Research Significance:
            {json.dumps(batch_info["significance"], ensure_ascii=False, indent=2)}

            Detailed Requirements:
            1. Ensure clear content structure with effective paragraphing
            2. Accurately understand and express the viewpoints of each literature
            3. Compare and analyze the viewpoints of different literature
            4. Identify the innovations and limitations of the research
            5. Reveal the research context and development trends
            6. Length: approximately 1000-1200 words
            
            Return in JSON format:
            {{
                "content": "Generated content",
                "citations": ["List of cited paper_ids"]
            }}
            """
        else:
            raise ValueError(f"Unsupported language: {lang}")

        return prompt
    
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

    def summarize_in_batches(self, query: str, documents: List[Dict], review_type, batch_size=10, lang="chinese") -> List[str]:
        partial_summaries = []
        all_citations = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i+batch_size]
            prompt, batch_citations = self._build_batch_prompt(query, batch_docs, review_type, lang=lang)
            all_citations.extend(batch_citations)
            summary_text = self.llm.generate(prompt, verbose=verbose)
            partial_summaries.append(summary_text)
        return partial_summaries, all_citations

    def _build_batch_prompt(self, query, batch_docs: List[Dict], review_type, lang="chinese") -> str:
        references_text = []
        batch_citations = []
        for paper_chunk in batch_docs:
            snippet = paper_chunk.content[:1500]
            citation_id = f"{paper_chunk.paper_id}-{paper_chunk.chunk_id}"
            batch_citations.append(citation_id)
            # ref_str = f"{snippet}...[{citation_id}]"
            ref_str = f"{snippet}...<chunk>{citation_id}</chunk>"
            references_text.append(ref_str)
        references_joined = "\n".join(references_text)

        if lang == "chinese":
            prompt = f"""
    你是一位学术研究助手。以下是一批文献内容片段，请为它们生成一个5000字以上的整合性的摘要。
    摘要内容要保证全面、连贯，尽可能引用文献片段原文，避免使用不存在于文献内容片段的观点。
    只参考有助于回答“{query}”{review_type_descriptions[review_type]}问题的片断，与问题无关的内容可以忽略。
    不需要摘要标题和章节，以列表形式给出符合给出的文献片段观点的描述。
    需要在对应信息后面标注引用，使用文献片段结尾的"<chunk>paper_id-chunk_id</chunk>"信息：

    {references_joined}

    请给出摘要：
    """
        elif lang == "english":
            prompt = f"""
    You are an academic research assistant. Here is a batch of paper content snippets. Please generate a comprehensive summary of over 5000 words for them.
    The summary should be comprehensive and coherent, and should try to quote the original text of the paper snippets as much as possible, avoiding the use of viewpoints that do not exist in the paper content snippets.
    Only refer to the fragments that help answer the question "{query}" {review_type_descriptions[review_type]}, and ignore content that is irrelevant to the question.
    No need for summary titles and chapters, give a description of the viewpoints of the paper fragments provided in list form.
    Citations should be marked after the corresponding information, using the "<chunk>paper_id-chunk_id</chunk>" information at the end of the paper snippet:
    
    {references_joined}
    
    Please provide a summary:
    """
        return prompt, batch_citations

########################
# 4. 多层聚合
########################
class MultiLevelAggregatorAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def iterative_aggregate(self, query: str, partial_summaries: List[str], review_type: ReviewType, max_chunk_size=2, lang="chinese") -> str:
        current_level = partial_summaries
        # while len(current_level) > 1:
        while len(current_level) > 3:
            next_level = []
            for i in range(0, len(current_level), max_chunk_size):
                chunk_summaries = current_level[i : i+max_chunk_size]
                input_text = "\n---\n".join(chunk_summaries)
                prompt = self._build_aggregation_prompt(query, input_text, review_type, lang=lang)
                retries = 0
                aggregated = ""
                while not "<chunk>" in aggregated and retries < 3:
                    aggregated = self.llm.generate(prompt, verbose=verbose)
                    retries += 1
                if "<chunk>" in aggregated:
                    next_level.append(aggregated)
            current_level = next_level
        # 最终只剩一个大摘要
        # return current_level[0]
        return "\n\n".join(current_level)

    # 如果有同一观点来自多个来源，可将引用合并在一起。
    # If the same viewpoint comes from multiple sources, the citations can be merged together.
    def _build_aggregation_prompt(self, query: str, input_text: str, review_type: ReviewType, lang="chinese") -> str:
        """
        聚合多个 partial_summary 的 Prompt，需保留引用标签。
        """

        if lang == "chinese":
            prompt = f"""
    请将以下多段摘要进行合并，写成一个更加全面、连贯的5000字以上的总结性文本，保持关键信息不丢失，并保留各段的引用标注（<chunk>paper_id-chunk_id</chunk>）。
    重点保留带有引用标注"<chunk>paper_id-chunk_id</chunk>"的文本片段的观点，可以直接引用，也可以根据需要进行整合。
    不需要摘要标题和章节，以列表形式给出摘要的观点。
    只参考有助于回答“{query}”{review_type_descriptions[review_type]}问题的片断，与问题无关的内容可以忽略。

    {input_text}

    请输出合并后的文本：
    """
        elif lang == "english":
            prompt = f"""
    Merge the following multiple summaries into a more comprehensive and coherent summary of over 5000 words, keeping the key information intact and retaining the reference tags (<chunk>paper_id-chunk_id</chunk>) of each section.
    Focus on retaining the viewpoints of text snippets with reference tags "<chunk>paper_id-chunk_id</chunk>", which can be directly quoted or integrated as needed. 
    No need for summary titles and chapters, provide the points of the summary in list form.
    Only refer to the fragments that help answer the question "{query}" {review_type_descriptions[review_type]}, and ignore content that is irrelevant to the question.
    
    {input_text}
    
    Please provide the merged text:
    """
        else:
            raise ValueError(f"Unsupported language: {lang}")

        return prompt

class StructuredWriterAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def write_review(self, query: str, final_summary: str, review_type: str, lang="chinese") -> str:
        if lang == "english":
            survey_prompt = f"""
    You are an academic writer. Please write a more comprehensive and coherent survey on "{query}" {review_type_descriptions[review_type]} based on and citing the following content of the viewpoint, ensuring that the key information from the original text is not lost.

    - A suitable title with H1 style. 
    - Only refer to the content that help answer the question "{query}" {review_type_descriptions[review_type]}, and ignore content that is irrelevant to the question.
    - Focus on retaining the viewpoints of text snippets with paper chunk reference tags "<chunk>paper_id-chunk_id</chunk>", which can be directly quoted or integrated as needed. 
    - At the end of the summary, list all paper chunk references in the "References" section in the format "[1]: paper_id-chunk_id"
    - Each paper chunk reference should be marked after the quoted content in the text, with the format "<sup>1</sup>".
    - Using English for both the title and content.
    - The summary must exceed 5000 words:
    
    {final_summary}
    
    Please provide the survey text:
    """
        elif lang == "chinese":
            survey_prompt = f"""
    您是一位学术作家。请根据以下观点内容撰写一份更全面且连贯的关于“{query}”的调查报告，确保不丢失原文中的关键信息。

    - 一个合适的标题，使用H1样式。
    - 仅参考有助于回答问题“{query}”的内容，并忽略与问题无关的内容。
    - 专注于保留文本片段中带有论文块引用标签"<chunk>paper_id-chunk_id</chunk>" 的观点，这些可以直接引用或按需整合。
    - 在总结末尾列出所有论文块引用，在“参考文献”部分以格式"[1]: paper_id-chunk_id"列出。
    - 每个论文块引用应在文本中被引述内容后标记。
    - 使用英语进行标题和内容书写。
    - 总结必须超过5000字：

    {final_summary}

    请提供调查文本：
    """
        else:
            raise ValueError(f"Unsupported language: {lang}")

        """
        根据不同的review_type对final_summary进行扩写，形成完整结构。
        """
        if lang == "chinese":
            common_requirements = """
    - 引用带标注"<chunk>paper_id-chunk_id</chunk>"的内容观点至少20个以上，并保留引用标记，将标记改写成"<sup>1</sup>" 的格式。
    - At the end of the summary, list all unique references in the "References" section in the format "[1]: paper_id-chunk_id"
    - 综述全文必须超过5000字:
            """
        elif lang == "english":
            common_requirements = """
    - Cite at least 20 viewpoints from the content marked with "<chunk>paper_id-chunk_id</chunk>", retaining the citation marks and rewriting them in the format "<sup>1</sup>".
    - At the end of the summary, list all unique references in the "References" section in the format "[1]: paper_id-chunk_id"
    - Using English for both the title and content.
    - The summary must exceed 5000 words:
            """

        if review_type == ReviewType.CONCEPT:
            prompt = survey_prompt

        elif review_type == ReviewType.STATUS:
            prompt = survey_prompt

        elif review_type == ReviewType.COMPARISON:
            prompt = survey_prompt

        elif review_type == ReviewType.TIMELINE:
            prompt = survey_prompt

        else:
            prompt = survey_prompt

        full_text = self.llm.generate(prompt, verbose=verbose)
        return full_text


########################
# 6. 引用校验 & 质量审校
########################
class CitationVerifierAgent:
    def verify_citations(self, text: str, original_docs: List[Dict]) -> str:
        """
        简化：暂不做深入校验，只做一个长度检测 & 占位。
        可扩展为: 逐条解析 "<chunk>paper_id-chunk_id</chunk>", 回溯原文做相似度检查等.
        """
        """
        1. 在最终文本中查找所有出现的 "<chunk>paper_id-chunk_id</chunk>"。
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


    def generate_review(self, query: str, review_type: ReviewType, output_file: str, lang="chinese") -> Dict:

        root_dir = f"outputs/{query}_{lang}"
        os.makedirs(root_dir, exist_ok=True)

        """生成完整的文献综述"""
        # 1. 扩展查询
        @cache_or_rebuild(cache_file=f"{root_dir}/expanded_queries.json")
        def do_expand_query(query: str, review_type: ReviewType) -> List[str]:
            """扩展查询"""
            return self.query_expander.expand_query(query, review_type, lang=lang)
        expanded_queries = do_expand_query(query, review_type)
        
        # 2. 检索内容
        @cache_or_rebuild(cache_file=f"{root_dir}/chunks.pkl")
        def do_retrieve_chunks(expanded_queries: List[str], min_papers=50) -> List[PaperChunk]:
            return self.retrieve_chunks(expanded_queries, min_papers=min_papers)
        chunks = do_retrieve_chunks(expanded_queries, min_papers=50)

        # @cache_or_rebuild(cache_file=f"{root_dir}/clustered_chunks.pkl")
        # def do_cluster_chunks(chunks: List[PaperChunk]) -> Dict:
        #     return self.retriever.cluster_chunks(chunks, lang=lang)
        # clustered_chunks = do_cluster_chunks(chunks)

        # 3) 分批摘要
        @cache_or_rebuild(cache_file=f"{root_dir}/partial_summaries.pkl")
        def do_partial_summaries(query: str, chunks: List[PaperChunk], review_type, batch_size=5) -> Tuple[List[str], List[str]]:
            return self.segment_summarizer.summarize_in_batches(query, chunks, review_type=review_type, batch_size=batch_size, lang=lang)
        partial_summaries, partial_citations = do_partial_summaries(query, chunks, review_type=review_type, batch_size=5)

        partial_citations_file = f"{root_dir}/partial_citations.txt"
        with open(partial_citations_file, "w", encoding="utf-8") as f:
            f.write("\n".join(partial_citations))

        # 4) 多层聚合
        @cache_or_rebuild(cache_file=f"{root_dir}/global_summary.pkl")
        def do_global_summary(query: str, partial_summaries: List[str], review_type: ReviewType, max_chunk_size=2) -> str:
            return self.aggregator.iterative_aggregate(query, partial_summaries, review_type=review_type, max_chunk_size=max_chunk_size, lang=lang)
        global_summary = do_global_summary(query, partial_summaries, review_type=review_type, max_chunk_size=2)

        global_summary = re.sub(r"\[\d+\]", "", global_summary)

        # 5) 结构化写作
        @cache_or_rebuild(cache_file=f"{root_dir}/structured_review.pkl")
        def do_write_review(query: str, global_summary: str, review_type: ReviewType) -> Dict:
            return self.writer.write_review(query, global_summary, review_type, lang=lang)
        structured_review = do_write_review(query, global_summary, review_type)

        review_content = structured_review

        review_content = re.sub(r"\[(\d+)\]\.", r"<sup>\1</sup>\.", review_content)

        # # 6) 引用校验
        # @cache_or_rebuild(cache_file=f"{root_dir}/verified_text.pkl")
        # def do_verify_citations(structured_review: Dict, chunks: List[PaperChunk]) -> Dict:
        #     return self.citation_verifier.verify_citations(structured_review, chunks)
        # verified_text = do_verify_citations(structured_review, chunks)

        # # 7) 质量审校
        # @cache_or_rebuild(cache_file=f"{root_dir}/review_content.pkl")
        # def do_quality_review(verified_text: str) -> str:
        #     return self.quality_agent.review_and_refine(verified_text)
        # review_content = do_quality_review(verified_text)

        process_references = True
        if process_references:
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

        # output_file = f"{root_dir}/review_{query}.md"
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
        


from intent_recognition import classify_intent_with_prompt
def do_summary(args):
    query = args.topic

    generate_params = {
        "temperature": 0.75,
        "top_p": 0.9,
        "max_tokens": 128 * 1000,
    }
    model_name = args.model_name
    api_key = args.token  
    llm_client = LLMClient(model_name=model_name, api_key=api_key, generate_params=generate_params)

    review_type = classify_intent_with_prompt(llm_client, query)

    print(f"Query: {query}\nReview Type: {review_type_descriptions[review_type]}\n")

    orchestrator = ReviewOrchestrator(llm_client=llm_client)
    
    logger.info(f"\n处理查询: {query}")
    result = orchestrator.generate_review(query, review_type, output_file=args.output_file, lang=args.lang)
    logger.debug(f"{result=}")


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="文献综述生成器")
    parser.add_argument("--topic", type=str, required=True, default=None, help="查询主题")
    parser.add_argument("--output_file", type=str, default="review.md", help="")
    parser.add_argument("--lang", type=str, default="chinese", choices=["chinese", "english"], help="语言")
    parser.add_argument("--model_name", type=str, default="glm-4-plus", help="模型名称")
    parser.add_argument("--token", type=str, default="618aa51377dab3e5435bf27793197955.r4dzbXJ6kaOZ8rA0", help="API Key")
    # parser.add_argument("--review_type", type=str, default=None, help="综述类型")
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

    do_summary(args)
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())