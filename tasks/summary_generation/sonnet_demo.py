from typing import List, Dict, Optional, Tuple, Set
import json
from dataclasses import dataclass
from enum import Enum
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import openai

from knowledge_base import kb_search_papers, kb_query_by_title_contain, kb_query_by_paper_id

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 初始化embedding模型
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
TOKEN_LIMIT = 4000  # LLM上下文窗口大小限制

# 定义综述类型枚举
class ReviewType(Enum):
    CONCEPT = "concept"  # 技术概念调研
    STATUS = "status"    # 研究现状
    COMPARISON = "comparison"  # 方法对比
    TIMELINE = "timeline"   # 技术脉络

# @dataclass
# class Paper:
#     paper_id: str
#     title: str
#     chunk_id: str
#     content: str

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

class ChunkManager:
    """管理文本块的分割和组织"""
    def __init__(self, max_tokens: int = TOKEN_LIMIT):
        self.max_tokens = max_tokens
        
    def split_papers_into_chunks(self, papers: List[PaperChunk]) -> List[List[PaperChunk]]:
        """将论文划分为适合LLM处理的chunks"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paper in papers:
            # 估算token数量 (简单估算: 单词数 * 1.3)
            paper_tokens = len(paper.content.split()) * 1.3
            
            if current_length + paper_tokens > self.max_tokens:
                if current_chunk:  # 保存当前chunk
                    chunks.append(current_chunk)
                current_chunk = [paper]
                current_length = paper_tokens
            else:
                current_chunk.append(paper)
                current_length += paper_tokens
        
        if current_chunk:  # 添加最后一个chunk
            chunks.append(current_chunk)
            
        return chunks

    def organize_by_topic(self, papers: List[PaperChunk]) -> Dict[str, List[PaperChunk]]:
        """按主题组织论文"""
        # 使用TF-IDF进行主题聚类
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([p.content for p in papers])
        
        # 提取主要主题词
        feature_names = vectorizer.get_feature_names_out()
        topic_papers = {}
        
        for i, paper in enumerate(papers):
            # 获取文档的主要主题词
            tfidf_scores = tfidf_matrix[i].toarray()[0]
            top_indices = tfidf_scores.argsort()[-3:][::-1]  # 取top3主题词
            topic = " ".join([feature_names[idx] for idx in top_indices])
            
            if topic not in topic_papers:
                topic_papers[topic] = []
            topic_papers[topic].append(paper)
            
        return topic_papers

class CitationManager:
    """管理引用的追踪和格式化"""
    def __init__(self):
        self.citations: Dict[str, Citation] = {}
        
    def add_citation(self, paper: PaperChunk, content: str) -> str:
        """添加引用并返回引用标记"""
        citation_key = f"{paper.paper_id}_{len(self.citations)}"
        self.citations[citation_key] = Citation(
            paper_id=paper.paper_id,
            chunk_id=paper.chunk_id,
            content=content,
            title=paper.title
        )
        return f"[{citation_key}]"
        
    def format_citations(self) -> str:
        """格式化所有引用"""
        formatted_citations = []
        for key, citation in sorted(self.citations.items()):
            formatted_citations.append(
                f"[{key}] {citation.title}. "
                f"(Paper ID: {citation.paper_id}, Chunk ID: {citation.chunk_id})"
            )
        return "\n".join(formatted_citations)
        
    def get_citations(self) -> List[Citation]:
        """获取所有引用"""
        return list(self.citations.values())

class ReviewAgent:
    """负责生成综述内容的智能体基类"""
    def __init__(self):
        self.review_type = None
        self.chunk_manager = ChunkManager()
        self.citation_manager = CitationManager()
        self.outline: List[str] = []
    
    async def generate_review(self, query: str, papers: List[PaperChunk]) -> str:
        """生成综述的基础实现"""
        # 1. 按主题组织论文
        topic_papers = self.chunk_manager.organize_by_topic(papers)
        
        # 2. 生成大纲
        self.outline = await self._generate_outline(query, topic_papers)
        
        #
class ConceptReviewAgent(ReviewAgent):
    """技术概念调研综述智能体"""
    def __init__(self):
        super().__init__()
        self.review_type = ReviewType.CONCEPT

    async def generate_review(self, query: str, papers: List[PaperChunk]) -> str:
        # 实现技术概念综述的生成逻辑
        structure = [
            "1. 概念定义与起源",
            "2. 核心技术组件",
            "3. 应用场景分析",
            "4. 技术优势与局限",
            "5. 未来发展方向"
        ]
        # TODO: 根据结构和papers生成综述内容
        return ""

class StatusReviewAgent(ReviewAgent):
    """研究现状综述智能体"""
    def __init__(self):
        super().__init__()
        self.review_type = ReviewType.STATUS

    async def generate_review(self, query: str, papers: List[PaperChunk]) -> str:
        structure = [
            "1. 研究背景与意义",
            "2. 关键技术突破",
            "3. 主要研究方向",
            "4. 存在的挑战",
            "5. 未来研究展望"
        ]
        # TODO: 实现研究现状综述生成逻辑
        return ""

class ComparisonReviewAgent(ReviewAgent):
    """方法对比综述智能体"""
    def __init__(self):
        super().__init__()
        self.review_type = ReviewType.COMPARISON

    async def generate_review(self, query: str, papers: List[PaperChunk]) -> str:
        structure = [
            "1. 问题背景",
            "2. 方法分类",
            "3. 典型方法详解",
            "4. 方法对比分析",
            "5. 最佳实践建议"
        ]
        # TODO: 实现方法对比综述生成逻辑
        return ""

class TimelineReviewAgent(ReviewAgent):
    """技术脉络综述智能体""" 
    def __init__(self):
        super().__init__()
        self.review_type = ReviewType.TIMELINE

    async def generate_review(self, query: str, papers: List[PaperChunk]) -> str:
        structure = [
            "1. 技术演进背景",
            "2. 关键里程碑",
            "3. 技术代际分析",
            "4. 重要突破点",
            "5. 发展趋势展望"
        ]
        # TODO: 实现技术脉络综述生成逻辑
        return ""

class QueryExpander_llm:
    """查询扩展智能体,负责将用户简短查询扩展为更全面的检索关键词"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def expand_query(self, query: str, review_type: ReviewType) -> List[str]:
        """扩展用户查询为多个相关查询"""
        prompt = f"""请基于以下用户查询,生成5-8个相关的检索关键词或短语,以帮助全面检索相关文献:
        用户查询: {query}
        综述类型: {review_type.value}
        要求:
        1. 关键词应该覆盖该主题的不同方面
        2. 包含该领域的专业术语
        3. 考虑近义词和相关概念
        4. 以JSON列表格式返回
        """
        
        response = self.llm.generate(prompt)
        return json.loads(response)

class QueryExpander:
    """查询扩展器：扩充原始查询以提高检索覆盖度"""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def expand_query(self, query: str) -> List[str]:
        """扩展原始查询"""
        expanded_queries = [query]
        
        # 1. 分词和预处理
        tokens = word_tokenize(query.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
        
        # 2. 生成关键词组合
        for i in range(len(tokens)):
            # 单个关键词
            expanded_queries.append(tokens[i])
            # 相邻词对
            if i < len(tokens) - 1:
                expanded_queries.append(f"{tokens[i]} {tokens[i+1]}")
                
        # 3. 添加领域特定扩展
        # 根据不同综述类型添加相关术语
        if "research status" in query.lower() or "现状" in query:
            expanded_queries.extend(["current progress", "recent advances", "challenges"])
        elif "development" in query.lower() or "发展" in query:
            expanded_queries.extend(["evolution", "timeline", "milestone"])
        elif "comparison" in query.lower() or "对比" in query:
            expanded_queries.extend(["comparison", "versus", "advantages"])
            
        return list(set(expanded_queries))

class PaperRetriever:
    """文献检索器"""
    def __init__(self, min_papers: int = 50):
        self.min_papers = min_papers
        self.query_expander = QueryExpander()
        self.embedding_model = EMBEDDING_MODEL

    async def retrieve_papers(self, query: str) -> List[PaperChunk]:
        """检索相关论文"""
        all_papers = {}  # paper_id -> PaperChunk
        
        # 1. 查询扩展
        expanded_queries = self.query_expander.expand_query(query)
        
        # 2. 多策略检索
        for expanded_query in expanded_queries:
            # 基于扩展查询检索
            results = kb_search_papers(expanded_query, top_k=50)
            for paper in results:
                if paper.paper_id not in all_papers:
                    all_papers[paper.paper_id] = paper
            
            # 基于标题检索
            title_results = kb_query_by_title_contain(expanded_query, top_k=30)
            for paper in title_results:
                if paper.paper_id not in all_papers:
                    all_papers[paper.paper_id] = paper
                    
            # 获取更多该论文的chunk
            for paper_id in all_papers:
                additional_chunks = kb_query_by_paper_id(paper_id, top_k=5)
                # 更新/添加chunk
                for chunk in additional_chunks:
                    if chunk.content not in {p.content for p in all_papers.values()}:
                        all_papers[f"{paper_id}_{len(all_papers)}"] = chunk
        
        # 3. 相关性排序
        ranked_papers = self._rank_papers(list(all_papers.values()), query)
        
        # 4. 确保最小论文数量
        if len(ranked_papers) < self.min_papers:
            # 通过引用网络扩展
            for paper in ranked_papers[:10]:  # 使用排名靠前的论文
                cited_papers = self._find_cited_papers(paper)
                for cited in cited_papers:
                    if cited.paper_id not in all_papers:
                        all_papers[cited.paper_id] = cited
            
            # 重新排序
            ranked_papers = self._rank_papers(list(all_papers.values()), query)
        
        return ranked_papers[:max(self.min_papers, len(ranked_papers))]

    def _find_cited_papers(self, paper: PaperChunk) -> List[PaperChunk]:
        """查找被引用的论文"""
        # 从文本中提取引用
        citations = re.findall(r'\[(.*?)\]', paper.content)
        cited_papers = []
        
        for citation in citations:
            # 尝试查找引用的论文
            results = kb_query_by_title_contain(citation, top_k=1)
            if results:
                cited_papers.extend(results)
                
        return cited_papers
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询中的关键词"""
        # TODO: 实现关键词提取逻辑
        return []
    
    def _rank_papers(self, papers: List[PaperChunk], query: str) -> List[PaperChunk]:
        """对论文进行相关性排序"""
        if not papers:
            return []
            
        # 1. 生成嵌入向量
        query_embedding = self.embedding_model.encode([query])[0]
        paper_embeddings = self.embedding_model.encode([p.content for p in papers])
        
        # 2. 计算相似度
        similarities = cosine_similarity([query_embedding], paper_embeddings)[0]
        
        # 3. 结合其他特征的评分
        paper_scores = []
        for i, paper in enumerate(papers):
            # 基础相似度分数
            score = similarities[i]
            
            # 标题匹配加权
            if any(term.lower() in paper.title.lower() for term in query.split()):
                score *= 1.2
                
            # 引用计数加权 (如果有的话)
            citation_count = len(re.findall(r'\[(.*?)\]', paper.content))
            score *= (1 + 0.1 * min(citation_count, 5))
            
            paper_scores.append((score, paper))
            
        # 4. 排序并返回论文
        ranked_papers = [p for _, p in sorted(paper_scores, reverse=True)]
        return ranked_papers

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
        chunks_with_analysis = self._analyze_chunks(relevant_chunks)
        
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
            result = self.llm.generate(prompt)
            parsed_result = json.loads(result)
            
            section_content.append(parsed_result["content"])
            all_citations.update(parsed_result["citations"])
            
            # 确保内容的连贯性
            if batch_idx > 0:
                transition_prompt = f"""请生成一个自然的过渡段落,连接以下两段内容:

                前一段结尾:
                {section_content[-2][-200:]}

                后一段开头:
                {section_content[-1][:200]}
                """
                transition = self.llm.generate(transition_prompt)
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
            
            要求返回JSON格式:
            {
                "key_points": ["关键点1", "关键点2"...],
                "methodology": "使用的方法",
                "findings": "主要发现",
                "significance": "重要性说明"
            }
            """
            
            analysis = json.loads(self.llm.generate(prompt))
            analyzed_chunks.append({
                "chunk": chunk,
                "analysis": analysis
            })
            
        return analyzed_chunks
    
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
        
        polished_content = self.llm.generate(polish_prompt)
        return polished_content

from  typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from enum import Enum

class ReviewType(Enum):
    CONCEPT = "concept_survey"  # 技术概念调研
    STATUS = "research_status"  # 研究现状
    COMPARISON = "method_comparison"  # 方法对比
    EVOLUTION = "tech_evolution"  # 技术演进

@dataclass
class PaperChunk:
    paper_id: str
    title: str
    chunk_id: str
    content: str

class QueryExpander:
    """查询扩展智能体,负责将用户简短查询扩展为更全面的检索关键词"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def expand_query(self, query: str, review_type: ReviewType) -> List[str]:
        """扩展用户查询为多个相关查询"""
        prompt = f"""请基于以下用户查询,生成5-8个相关的检索关键词或短语,以帮助全面检索相关文献:
        用户查询: {query}
        综述类型: {review_type.value}
        要求:
        1. 关键词应该覆盖该主题的不同方面
        2. 包含该领域的专业术语
        3. 考虑近义词和相关概念
        4. 以JSON列表格式返回
        """
        
        response = self.llm.generate(prompt)
        return json.loads(response)

class ContentRetriever:
    """内容检索智能体,负责从文献库检索相关内容"""
    
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
    
    def retrieve_by_queries(self, queries: List[str], min_papers=50) -> List[PaperChunk]:
        """基于多个查询检索文献内容"""
        all_chunks = []
        papers_seen = set()
        
        for query in queries:
            chunks = kb_search_papers(query)
            for chunk in chunks:
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
        for chunk in chunks:
            prompt = f"""请为以下文献片段生成3个主题标签,以逗号分隔:
            标题: {chunk.title}
            内容: {chunk.content[:500]}  # 使用前500字符以节省token
            """
            topics = self.llm.generate(prompt).strip().split(",")
            chunk_topics[chunk.chunk_id] = [t.strip() for t in topics]
        
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
        4. 以JSON格式返回,包含章节标题和子标题
        """
        
        response = self.llm.generate(prompt)
        return json.loads(response)
    
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
            summary = self.llm.generate(prompt)
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
            
        context = self.llm.generate(template)
        return context

class ContentGenerator:
    """内容生成智能体,负责生成每个章节的具体内容"""
    
    def __init__(self, llm_client, max_tokens=2000):
        self.llm = llm_client
        self.max_tokens = max_tokens
        
    def generate_section(self, section_title: str, 
                        relevant_chunks: List[PaperChunk]) -> Tuple[str, List[str]]:
        """生成单个章节的内容和引用"""
        chunks_batches = self._split_chunks(relevant_chunks)
        section_content = []
        citations = []
        
        for batch in chunks_batches:
            prompt = f"""基于以下参考文献内容,为章节'{section_title}'撰写一段内容:
            参考内容: {self._format_chunks(batch)}
            
            要求:
            1. 内容需要融合多篇文献的观点
            2. 确保行文流畅,逻辑清晰
            3. 准确标注引用来源
            4. 以JSON格式返回,包含内容文本和引用列表
            """
            
            response = self.llm.generate(prompt)
            result = json.loads(response)
            section_content.append(result["content"])
            citations.extend(result["citations"])
            
        return "\n".join(section_content), citations
    
    def _split_chunks(self, chunks: List[PaperChunk]) -> List[List[PaperChunk]]:
        """将chunks分批以适应LLM上下文窗口"""
        current_batch = []
        current_size = 0
        batches = []
        
        for chunk in chunks:
            chunk_size = len(chunk.content)
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

class ReviewQuality:
    """综述质量评估指标"""
    def __init__(self):
        self.metrics = {
            'coverage': 0.0,  # 文献覆盖度
            'structure': 0.0,  # 结构完整性
            'coherence': 0.0,  # 内容连贯性
            'citation': 0.0,   # 引用准确性
            'novelty': 0.0     # 见解新颖性
        }
    
    def update(self, metric: str, score: float):
        if metric in self.metrics:
            self.metrics[metric] = score
    
    def get_overall_score(self) -> float:
        weights = {
            'coverage': 0.25,
            'structure': 0.2,
            'coherence': 0.2,
            'citation': 0.25,
            'novelty': 0.1
        }
        return sum(self.metrics[m] * weights[m] for m in self.metrics)

class ReviewOrchestrator:
    """系统协调器,负责整体流程控制"""
    
    def __init__(self, llm_client):
        self.query_expander = QueryExpander(llm_client)
        self.retriever = ContentRetriever()
        self.outline_generator = OutlineGenerator(llm_client)
        self.content_generator = ContentGenerator(llm_client)
        
    def evaluate_quality(self, review_content: Dict, original_chunks: List[PaperChunk]) -> ReviewQuality:
        """评估综述质量"""
        quality = ReviewQuality()
        
        # 1. 评估文献覆盖度
        cited_papers = set()
        for section in review_content["content"]:
            for citation in section["citations"]:
                cited_papers.add(citation)
        coverage = len(cited_papers) / len({chunk.paper_id for chunk in original_chunks})
        quality.update('coverage', coverage)
        
        # 2. 评估结构完整性
        prompt = f"""评估以下综述的结构完整性(0-1分):
        类型: {review_content["review_type"]}
        大纲: {[section["title"] for section in review_content["content"]]}
        
        评分标准:
        1. 结构是否完整(引言、主体、总结等)
        2. 章节层次是否合理
        3. 逻辑关系是否清晰
        4. 是否符合该类型综述的标准结构
        
        只返回分数数字
        """
        structure_score = float(self.llm.generate(prompt))
        quality.update('structure', structure_score)
        
        # 3. 评估内容连贯性
        all_content = "\n".join(section["content"] for section in review_content["content"])
        prompt = f"""评估以下综述的内容连贯性(0-1分):
        {all_content[:2000]}  # 评估前2000字
        
        评分标准:
        1. 段落之间是否衔接自然
        2. 论述是否层次分明
        3. 是否存在重复或矛盾
        4. 专业术语使用是否准确一致
        
        只返回分数数字
        """
        coherence_score = float(self.llm.generate(prompt))
        quality.update('coherence', coherence_score)
        
        # 4. 评估引用准确性
        # 随机抽查5个引用
        import random
        sample_citations = random.sample(list(cited_papers), min(5, len(cited_papers)))
        citation_scores = []
        
        for paper_id in sample_citations:
            # 获取原文内容
            paper_chunks = [c for c in original_chunks if c.paper_id == paper_id]
            original_content = "\n".join(c.content for c in paper_chunks)
            
            # 找到综述中引用该文献的段落
            citing_passages = []
            for section in review_content["content"]:
                if paper_id in section["citations"]:
                    citing_passages.append(section["content"])
            
            prompt = f"""评估引用准确性(0-1分):
            原文: {original_content[:1000]}
            综述中的引用: {' '.join(citing_passages)[:500]}
            
            评分标准:
            1. 是否准确理解原文观点
            2. 是否存在断章取义
            3. 引用位置是否恰当
            4. 是否有效支持论述
            
            只返回分数数字
            """
            score = float(self.llm.generate(prompt))
            citation_scores.append(score)
        
        quality.update('citation', sum(citation_scores) / len(citation_scores))
        
        # 5. 评估见解新颖性
        prompt = f"""评估以下综述的见解新颖性(0-1分):
        综述内容: {all_content[:2000]}
        
        评分标准:
        1. 是否提出新的观点或视角
        2. 是否有效总结研究趋势
        3. 是否深入分析问题本质
        4. 是否提供有价值的未来展望
        
        只返回分数数字
        """
        novelty_score = float(self.llm.generate(prompt))
        quality.update('novelty', novelty_score)
        
        return quality

    def generate_review(self, query: str, review_type: ReviewType) -> Dict:
        """生成完整的文献综述"""
        # 1. 扩展查询
        expanded_queries = self.query_expander.expand_query(query, review_type)
        
        # 2. 检索内容
        chunks = self.retriever.retrieve_by_queries(expanded_queries)
        if len(chunks) < 50:
            print(f"Warning: Only found {len(chunks)} relevant chunks, less than minimum requirement of 50")
        
        clustered_chunks = self.retriever.cluster_chunks(chunks)
        
        # 3. 生成大纲
        outline = self.outline_generator.generate_outline(
            query, review_type, chunks[:10]  # 使用部分chunks生成大纲
        )
        
        # 4. 按章节生成内容
        review_content = []
        all_citations = []
        total_words = 0
        
        for section in outline:
            section_chunks = self._get_relevant_chunks(
                section["title"], 
                clustered_chunks
            )
            
            # 确保每个章节至少有一定数量的相关文献
            if len(section_chunks) < 10:
                print(f"Warning: Section '{section['title']}' only has {len(section_chunks)} relevant chunks")
                # 补充检索
                additional_chunks = kb_search_papers(
                    f"{query} {section['title']}", 
                    top_k=20
                )
                section_chunks.extend([c for c in additional_chunks if c.chunk_id not in {x.chunk_id for x in section_chunks}])
            
            content, citations = self.content_generator.generate_section(
                section["title"],
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
            need_optimization = self.llm.generate(prompt).strip().lower() == "yes"
            
            if need_optimization:
                # 使用更多上下文重新生成
                content, citations = self.content_generator.generate_section(
                    section["title"],
                    section_chunks,
                    detailed_requirements=True
                )
            
            review_content.append({
                "title": section["title"],
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
        
        # 评估质量
        quality = self.evaluate_quality(result, chunks)
        result["metrics"] = quality.metrics
        result["overall_score"] = quality.get_overall_score()
        
        return result
        
    def _expand_section_content(self, title: str, existing_content: str, 
                              clustered_chunks: Dict[str, List[PaperChunk]]) -> Tuple[str, List[str]]:
        """扩展章节内容"""
        prompt = f"""分析以下章节内容,指出需要补充的方面:
        标题: {title}
        现有内容: {existing_content}
        
        以逗号分隔的关键词形式返回需要补充的方面
        """
        aspects_to_expand = self.llm.generate(prompt).strip().split(",")
        
        additional_chunks = []
        for aspect in aspects_to_expand:
            # 检索相关内容
            relevant_chunks = kb_search_papers(
                f"{title} {aspect}",
                top_k=10
            )
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
        section_keywords = set(self.llm.generate(prompt).strip().split(","))
        
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
            cluster_keywords = set(self.llm.generate(prompt).strip().split(","))
            
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
                    score = float(self.llm.generate(prompt))
                    chunk_scores.append((chunk, score))
                
                # 添加相关度较高的chunks
                sorted_chunks = [c for c, s in sorted(chunk_scores, key=lambda x: x[1], reverse=True)
                               if s > 0.3]  # 相关度阈值
                relevant_chunks.extend(sorted_chunks)
        
        return relevant_chunks

# async def main():
#     """主函数"""
#     # 示例查询
#     queries = [
#         ("损失函数的发展与应用", ReviewType.CONCEPT),
#         ("Text2SQL研究现状与挑战", ReviewType.STATUS),
#         ("大模型规划能力提升方法对比", ReviewType.COMPARISON),
#         ("多模态大模型技术发展路线", ReviewType.TIMELINE)
#     ]
    
#     orchestrator = ReviewOrchestrator()
    
#     for query, review_type in queries:
#         print(f"\n处理查询: {query}")
#         try:
#             review, citations = await orchestrator.generate_review(query, review_type)
#             print(f"\n生成的综述 ({len(review)} 字符):")
#             print(f"{review[:500]}...")
#             print(f"\n使用的引用数量: {len(citations)}")
#         except Exception as e:
#             print(f"处理失败: {e}")
#             continue

# # 使用示例
# async def main():
#     orchestrator = ReviewOrchestrator()
    
#     # 示例查询
#     queries = [
#         "损失函数的发展与应用",  # 概念综述
#         "Text2SQL研究现状与挑战",  # 现状综述
#         "大模型规划能力提升方法对比",  # 对比综述
#         "多模态大模型技术发展路线"  # 脉络综述
#     ]
    
#     for query in queries:
#         review_content, citations = await orchestrator.generate_review(query)
#         print(f"Query: {query}")
#         print("Review content:", review_content[:200], "...")
#         print(f"Number of citations: {len(citations)}\n")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

# 使用示例
if __name__ == "__main__":
    # 1. 初始化LLM客户端(需要自行实现)
    class SimpleLLMClient:
        def generate(self, prompt: str) -> str:
            """简单的LLM接口封装"""
            # 实际项目中需要对接具体的LLM服务
            pass
    
    llm = SimpleLLMClient()
    orchestrator = ReviewOrchestrator(llm)
    
    # 2. 测试不同类型的综述生成
    
    # 2.1 技术概念综述
    concept_query = "损失函数在深度学习中的应用与发展"
    concept_review = orchestrator.generate_review(
        concept_query,
        ReviewType.CONCEPT
    )
    print(f"\n概念综述完成: {concept_review['total_words']}字")
    print(f"质量评分: {concept_review['overall_score']:.2f}")
    
    # 2.2 研究现状综述
    status_query = "Text2SQL研究现状与挑战"
    status_review = orchestrator.generate_review(
        status_query,
        ReviewType.STATUS
    )
    print(f"\n现状综述完成: {status_review['total_words']}字")
    print(f"质量评分: {status_review['overall_score']:.2f}")
    
    # 2.3 方法对比综述
    comparison_query = "大模型规划能力提升方法对比研究"
    comparison_review = orchestrator.generate_review(
        comparison_query,
        ReviewType.COMPARISON
    )
    print(f"\n对比综述完成: {comparison_review['total_words']}字")
    print(f"质量评分: {comparison_review['overall_score']:.2f}")
    
    # 2.4 技术演进综述
    evolution_query = "多模态大模型的技术发展与未来趋势"
    evolution_review = orchestrator.generate_review(
        evolution_query,
        ReviewType.EVOLUTION
    )
    print(f"\n演进综述完成: {evolution_review['total_words']}字")
    print(f"质量评分: {evolution_review['overall_score']:.2f}")
    
    # 3. 结果分析示例
    def analyze_review_quality(review: Dict):
        """分析综述质量"""
        print(f"\n综述主题: {review['query']}")
        print(f"综述类型: {review['review_type']}")
        print(f"总字数: {review['total_words']}")
        print("\n质量指标:")
        for metric, score in review['metrics'].items():
            print(f"- {metric}: {score:.2f}")
        print(f"\n引用文献数: {len(review['citations'])}")
        print("\n章节统计:")
        for section in review['content']:
            print(f"- {section['title']}: {section['word_count']}字, "
                  f"{len(section['citations'])}篇引用")
    
    # 分析示例综述
    print("\n========= 综述质量分析 =========")
    analyze_review_quality(concept_review)