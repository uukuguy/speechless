"""
指令解析与综述类型识别
文献知识库检索
长文自动写作（引用索引）
质量审校与最终输出

## 一、系统整体架构与流程
本系统从用户输入“综述主题与类型”开始，经过多智能体协作，完成对文献知识库检索、文本聚合、内容结构化生成、索引溯源等步骤，最终生成长文献综述。主要包含以下几个关键角色（或智能体）：

1. 指令解析与意图识别智能体（Instruction Parser / Intent Agent）

- 功能：识别用户输入对应的综述类型 (如：技术概念调研综述、方向研究现状综述、多方法对比分析综述、技术方法脉络综述等)，并抽取关键信息（主题概念、研究方向、方法名称、时间范围等）。
- 输出：确定综述类型、检索关键词、预期输出结构等。

2. 检索与外部知识增强智能体（Retrieval Agent）

- 功能：根据指令解析与意图识别智能体传来的检索关键词、主题、等信息调用文献知识库接口（如 kb_search_papers，kb_query_by_title 等）检索到最相关的文献切片。
- 输出：与主题相关的文献切片信息（包含 paper_id, title, chunk_id, content 等），一般可检索到数量不少于 50 篇（或更多）相关文献片段，以丰富后续的综述生成。

3. 结构化长文献综述生成智能体（Summarization Agent / Writer Agent）

- 功能：基于获取到的检索结果进行多轮分析、摘要、写作，产生结构化的长文献综述草稿。
要求：
    - 能够按照综述类型输出合理的章节/段落结构（例如章节标题、技术脉络、对比分析等）；
    - 内容细节需引用检索到的论文片段；
    - 内容篇幅不低于 2000 字；
    - 对于引用内容提供句子或段落级别的索引（如在段落或句子末附上 [paper_id-chunk_id] 或 [title-chunk_id]）。

4. 质量评估与审校智能体（Quality & Proofreading Agent）

- 功能：对生成的结构化综述做校验与优化，包括：
    - 内容连贯性、合规性、上下文逻辑正确性；
    - 格式、排版、引用格式等统一性检查；
    - 是否满足“不少于 2000 字”的长度要求。
- 输出：最终可交付的结构化长文献综述文本。

最终，系统将生成的结构化长文献综述返回给用户，同时提供可回溯引用索引。

## 二、主要功能流程图 (简要)

```mermaid
flowchart LR
    A[用户输入综述指令] --> B[指令解析与意图识别智能体]
    B --> C[检索与外部知识增强智能体]
    C --> C1[检索调用:\n kb_search_papers(), kb_query_by_...\n 等接口]
    C1 --> C2[检索结果合并 & 去重]
    C2 --> D[结构化长文献综述生成智能体]
    D --> E[质量评估与审校智能体]
    E --> F[最终输出给用户]
```

五、关键环节说明

1. 多智能体的拆分：

为了更加灵活和可扩展，每个智能体负责一个相对独立的功能（解析、检索、总结、质量审校），在工程落地中可以根据需要拆分或合并。
检索策略：

在示例中，为简化仅用 kb_search_papers 与关键词做检索。在真实系统中，可根据需求组合使用 kb_query_by_paper_id、kb_query_by_title、kb_query_by_title_contain、kb_query_by_chunk_contain 等更加精细的检索方式。

2. 文献片段合并与去重：

在 RetrievalAgent 中，对检索到的文献切片进行合并去重，以免同一论文片段重复引用，或浪费 token 资源。
结构化写作的 Prompt：

在 SummarizationAgent 中，根据不同综述类型（如技术概念调研、研究方向综述、多方法对比、技术脉络演进等）设计不同的 Prompt。
Prompt 中加入参考文献片段并指导模型在输出时带上 [paper_id-chunk_id] 等索引，实现“句子或段落级别”回溯。

3. 字数与质量保障：

在 QualityAgent 中设置最低字数限制，若生成文本不足 2000 字，则进行补充写作或合并其它引用信息。
可进一步加入自动化测试脚本，对生成文本的引用索引、逻辑连贯等进行检测，并做相应修正。

4. 拓展性：

若后续要支持更多类型的综述、更多元的检索策略或更高级的审校逻辑，可以通过拓展对应智能体的类或方法来完成，无需大幅改动整体架构。

重点针对以下两个难点进行阐述：

文献数量和长度超出单次大模型调用限制：需要分层/多段方式处理大规模文献，并将中间总结逐步聚合。
引用标注需要做到句子或段落级别的索引溯源，并且要保证引用内容的准确性和可追溯性。

分块检索 + 分段汇总 + 逐级聚合 + 引用校验

对海量论文的长篇综述，单次将所有检索到的文献片段直接拼接到大模型的 Prompt 往往不现实，因为长度（上下文窗口）可能远远超出大模型的输入上限。为此，可以考虑分层/分块策略，将上百篇、上千篇文献片段分批处理，每批仅对若干文献进行深入摘要，然后再对摘要进行聚合、再总结，从而形成层级化的长文档摘要结构。

同时，引用标注可在最初提取和每次局部总结时，就将对应的paper_id-chunk_id标记到文本中，使最终总结依然能追溯到原始内容。

## 详细流程设计

2.1 模块划分
1. 指令解析与意图识别智能体 (Instruction Parser)

- 输入：用户的自然语言问题或综述请求。
- 输出：识别出的综述类型（概念调研/方向综述/对比分析/技术脉络…），以及核心关键词、时间范围或其他检索要求等。

2. 检索与初步过滤智能体 (Retrieval Agent)

- 输入：由解析器输出的关键词/主题信息。
- 输出：在文献知识库中检索到的所有满足或最相关的文献切片（可能包含数百篇甚至更多）。同时进行去重、基础过滤（如论文发布时间、是否符合主题等）。

3. 文档分块与分批Summarization智能体 (Segment Summarizer Agent)

- 输入：海量待处理的文献切片列表。
- 输出：将文献切片分成若干“批次”或“聚类”，对每一批进行小规模汇总（Partial Summaries），并在文本中保留对应的引用索引（例如 [paper_id-chunk_id]）。
- 必要性：因为大模型一次性输入有限，需要分批处理，再将分批结果进行二次或多次聚合。

4. 多轮聚合智能体 (Multi-Level Aggregation Agent)

- 输入：上一步各批的Partial Summaries（可能有几十段甚至上百段）。
- 输出：在保证引用索引不丢失的前提下，将多个Partial Summaries合并为更高层次的Summary，反复迭代直到最终只剩1~2段大规模的全局汇总文本，并且依然在相应语句/段落处保留引用索引。

5. 结构化综述生成智能体 (Structured Writer Agent)

- 输入：最终聚合后的全局汇总文本 + 用户的综述类型（概念调研/对比分析等）
- 输出：按照预先定义的结构化章节，将聚合后的文本扩写为一篇完整的、具有清晰大纲和逻辑脉络的长文献综述，内容不少于 2000 字；文中引用位置需遵循一定的规则（如在文末或句末标注 [paper_id-chunk_id]）。

6. 引用准确性校验与质量审校智能体 (Citation Verifier & QA Agent)

- 输入：完整的长文献综述初稿。
- 功能：
    - 验证：检验文中引用 [paper_id-chunk_id] 与上下文内容的匹配度，避免“错误地借用了不相关片段”或“丢失了本该引用的片段”。
    - 修订：若发现引用位置不正确，或者需要补充/删除引用，进行二次修改；并且再次校验字数、段落结构、逻辑连贯性等。
    - 输出：最终的高质量、带引用标记的结构化综述。

## 分批处理与多轮聚合的示例流程

为解决“文献太多、单次超长”的问题，以下是一个常见的分层总结策略（也可视为分段多次调用大模型的一种流水线）：

1. 输入：检索到的文献切片列表 documents，数目可能高达几百上千。
2. 分批：将文献切片按照一定策略（如时间顺序、主题相似度聚类、随机分组）分成 N 批，每批包含 M 条切片（例如每批 10~20 篇）。
3. 批次小摘要 (Partial Summaries)：对每批文献切片调用大模型或使用其他摘要方法进行局部总结。在此步骤产生的文本称为partial_summary_i。
    - 在partial_summary_i中，需要在合适的句子/段落后面加上对应的 [paper_id-chunk_id] 引用，使得后续聚合仍能追溯到原始文献切片。
4. 第一层聚合：假设我们有 N 个 partial_summary_i，如果所有小摘要总长度仍超过大模型的处理上限，则需要再次分批聚合：
    - 将多个 partial_summary_i 组成第二层小批次，继续二次Summarization，输出更高层次的摘要 meta_summary_j，在其中依然保留或合并引用信息。
5. 多层循环：重复以上聚合过程，直到聚合出一个或者少数几个能够放进大模型上下文的终级摘要。
6. 生成最终综述：再调用一次“结构化综述写作”大模型，将终级摘要扩写成完整结构的长文献综述。
7. 引用校验与质量审校：最终输出的长文献综述需要经过引用标记校验、逻辑连贯性审校等环节，确保综述内容的准确性和质量。
通过以上多轮摘要，可逐级消化海量文本，而不必一次性输入全部文献，有效绕过大模型的上下文限制。

"""

"""
引用溯源与准确性保证
2.3.1 引用标注策略
在每次“分批 Summarization”时嵌入引用信息

例如，在对某一批文献切片 [doc1, doc2, ... docM] 进行小总结时，对于最终产出的文本 partial_summary_i，如果某段话主要来自 doc2，则在该段话末标注 [paper_id-chunk_id]。
若是聚合了多篇文献的意见，可能需要并列多个标注，如 [paperA-chunk5; paperB-chunk1; paperC-chunk2]。
在后续聚合阶段保留或合并引用

当二次Summarization时，输入往往是上一层的 partial_summary，它里面已经带好引用索引。
大模型输出新的摘要文本时，需要引导它“保留原有引用索引”，并在多篇文献的合并处合并引用。
可以在提示中明确告诉模型：
“请在合并后的文本中，保留并适当合并原文本中的所有引用标注。如果一句话综合了多个来源，请在句尾以分号分隔列出引用ID。”

最终成文时的引用格式

在最终的长文献综述里，仍建议采用段落/句子末的 [paper_id-chunk_id] 形式。
若需要格式更学术化，也可以在文本中写 [1] [2] ...，对应在文后给出全引用表，但依然需要能从 [1] 找到对应的 paper_id-chunk_id。
2.3.2 减少“虚假引用”或“错误引用”的措施
大模型提示约束：在Prompt中反复强调：

“不可随意编造引用标记。”
“所有引用标记必须来自输入文本的原有 [paper_id-chunk_id] 列表，且仅在合适位置出现。”
自动引用验证（Citation Verification）

在引用验证智能体中，可以对生成结果进行检测：
查找每个引用标记 [paper_id-chunk_id] 所在的句子/段落；
根据 chunk_id 回溯到原始文献切片，检查关键的事实一致性（例如主要结论、指标、观点是否与文献内容相符）。
如果出现明显不一致（如原文并未支持此论点），可给出警告或自动移除/修改该引用。
人工或半自动检查：对于高价值、高风险的引用，最好还要有一定的人工审校流程，毕竟目前大模型仍有可能出现“幻觉”或“张冠李戴”的情况。
"""

############################################
# file: literature_review_system.py
############################################

from typing import List, Dict

########################
# 0. 伪造的知识库接口 (示例)
########################
def kb_search_papers(query: str, top_k: int = 30) -> List[Dict]:
    """
    根据文本查询搜索论文片段(示例化返回)。实际项目中可连数据库或向量检索。
    返回: [{"paper_id":..., "title":..., "chunk_id":..., "content":...}, ...]
    """
    # 这里为了演示，直接返回mock数据
    return [
        {
            "paper_id": "paperA",
            "title": "Loss Function Innovations",
            "chunk_id": "1",
            "content": "损失函数在深度学习训练中的重要性...[paperA-1]"
        },
        {
            "paper_id": "paperB",
            "title": "Large Language Models for Planning",
            "chunk_id": "2",
            "content": "提升大模型规划能力的常见思路包括... [paperB-2]"
        },
        # ...更多mock
    ][:top_k]

########################
# 1. 指令解析与意图识别
########################
class InstructionParser:
    """
    通过微调好的分类模型，预测意图，并抽取关键主题/关键词。
    """
    def __init__(self, intent_classifier):
        self.intent_classifier = intent_classifier

    def parse(self, user_input: str) -> dict:
        # 1) 用分类器预测intent
        predicted_intent = self.intent_classifier.predict_intent(user_input)

        # 2) 简单关键词抽取(可改为更复杂的NLP方法)
        #   这里仅演示, 若intent=concept, 取前几个词作为keywords
        tokens = user_input.split()
        keywords = tokens[:2]  # 简陋做法

        return {
            "review_type": predicted_intent,
            "keywords": keywords,
            "raw_input": user_input
        }

########################
# 2. 检索与外部知识增强
########################

class RetrievalAgent:
    """
    检索与外部知识增强智能体
    """
    def retrieve_documents(self, parse_result: dict, min_papers: int = 50) -> List[Dict]:
        """
        根据解析结果中的 keywords 调用知识库检索相关文献切片。
        默认希望检索到不少于 min_papers 篇文献切片，
        并进行去重或合并操作。
        """
        keywords = parse_result["keywords"]
        
        # 简化策略：先用 kb_search_papers 对每个关键词检索，再合并去重
        all_chunks = []
        for kw in keywords:
            result = kb_search_papers(kw, top_k=min_papers)
            all_chunks.extend(result)
        
        # 合并去重 (简单按 paper_id + chunk_id 去重)
        unique_chunks_dict = {}
        for chunk in all_chunks:
            uid = f"{chunk['paper_id']}_{chunk['chunk_id']}"
            if uid not in unique_chunks_dict:
                unique_chunks_dict[uid] = chunk
        
        # 返回去重后的文献切片列表
        unique_chunks = list(unique_chunks_dict.values())
        
        return unique_chunks


########################
# 3. 分批摘要
########################
def chat_with_model(prompt: str) -> str:
    """
    伪代码：对接大语言模型的函数
    """
    # 真实项目中可使用 OpenAI / HuggingFace API
    return f"【LLM摘要】\n{prompt}\n"

class SegmentSummarizerAgent:
    def summarize_in_batches(self, documents: List[Dict], batch_size=10) -> List[str]:
        partial_summaries = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i+batch_size]
            prompt = self._build_batch_prompt(batch_docs)
            summary_text = chat_with_model(prompt)
            partial_summaries.append(summary_text)
        return partial_summaries

    def _build_batch_prompt(self, batch_docs: List[Dict]) -> str:
        references_text = []
        for doc in batch_docs:
            snippet = doc["content"][:300]
            ref_str = f"{snippet}...[{doc['paper_id']}-{doc['chunk_id']}]"
            references_text.append(ref_str)
        references_joined = "\n".join(references_text)

#         prompt = f"""
# 下面是一批论文片段，请你总结它们的关键信息并保留引用标注（[paper_id-chunk_id]），不要超过1500字。
# {references_joined}
# """
        prompt = f"""
你是一位学术研究助手。以下是一批文献内容片段，请为它们生成一个整合性的摘要，不超过1500字。
需要在对应信息后面标注引用，使用文献片段结尾的[paper_id-chunk_id]信息：

{references_joined}

请给出摘要：
"""
        return prompt

########################
# 4. 多层聚合
########################
class MultiLevelAggregatorAgent:
    def iterative_aggregate(self, partial_summaries: List[str], max_chunk_size=3) -> str:
        current_level = partial_summaries
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), max_chunk_size):
                chunk_summaries = current_level[i : i+max_chunk_size]
                input_text = "\n---\n".join(chunk_summaries)
                prompt = self._build_aggregation_prompt(input_text)
                aggregated = chat_with_model(prompt)
                next_level.append(aggregated)
            current_level = next_level
        # 最终只剩一个大摘要
        return current_level[0]

#     def _build_aggregation_prompt(self, input_text: str) -> str:
#         prompt = f"""
# 请将以下多段摘要进行合并，写成一个更全面的总结性文本，并保留/合并引用标注：
# {input_text}
# """
#         return prompt
    def _build_aggregation_prompt(self, input_text: str) -> str:
        """
        聚合多个 partial_summary 的 Prompt，需保留引用标签。
        """
        prompt = f"""
请将以下多段摘要进行合并，写成一个更加精炼、连贯的总结性文本，保持关键信息不丢失，并保留各段的引用标注（[paper_id-chunk_id]）。
如果有同一观点来自多个来源，可将引用合并在一起。

{input_text}

请输出合并后的文本：
"""
        return prompt

########################
# 5. 结构化写作
########################
class StructuredWriterAgent:
    def write_review(self, final_summary: str, review_type: str) -> str:
        """
        根据不同的review_type对final_summary进行扩写，形成完整结构。
        """
        if review_type == "concept":
            prompt = f"""
你是一位学术写作者。根据下列综合摘要，写一篇关于该技术概念的调研综述，包括：
1) 概念定义
2) 分类或子领域
3) 应用场景
4) 主要挑战
5) 未来展望

不少于2000字，并保留原有的引用标注 [paper_id-chunk_id]:

{final_summary}
"""
        elif review_type == "direction":
            prompt = f"""
写一篇关于某研究方向的综述，包括研究背景、现状、挑战和未来方向，不少于2000字，保留引用:
{final_summary}
"""
        elif review_type == "comparison":
            prompt = f"""
写一篇对多种方法进行对比分析的综述，包括核心思想、优缺点、性能对比与改进方向，不少于2000字，保留引用:
{final_summary}
"""
        elif review_type == "evolution":
            prompt = f"""
写一篇技术方法的发展脉络综述，包括早期研究、关键里程碑、新进展和未来趋势，不少于2000字，保留引用:
{final_summary}
"""
        else:
            prompt = f"""
请写一篇通用综述，不少于2000字，保留引用：
{final_summary}
"""
        full_text = chat_with_model(prompt)
        return full_text

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
        if len(text) < 2000:
            # 若不够长, 进行简单补充
            text += "\n\n(自动补充内容以满足2000字要求) " + "延伸讨论..."*100
        return text

########################
# 7. 总统筹的System
########################
class LiteratureReviewSystem:
    def __init__(self, intent_classifier):
        self.parser = InstructionParser(intent_classifier)
        self.retriever = RetrievalAgent()
        self.segment_summarizer = SegmentSummarizerAgent()
        self.aggregator = MultiLevelAggregatorAgent()
        self.writer = StructuredWriterAgent()
        self.citation_verifier = CitationVerifierAgent()
        self.quality_agent = QualityAgent()

    def generate_review(self, user_input: str) -> str:
        # 1) 意图解析
        parse_result = self.parser.parse(user_input)
        # 2) 文献检索
        docs = self.retriever.retrieve_documents(parse_result)
        if not docs:
            return "未找到相关文献"
        # 3) 分批摘要
        partial_summaries = self.segment_summarizer.summarize_in_batches(docs, batch_size=5)
        # 4) 多层聚合
        global_summary = self.aggregator.iterative_aggregate(partial_summaries, max_chunk_size=2)
        # 5) 结构化写作
        structured_review = self.writer.write_review(global_summary, parse_result["review_type"])
        # 6) 引用校验
        verified_text = self.citation_verifier.verify_citations(structured_review, docs)
        # 7) 质量审校
        final_text = self.quality_agent.review_and_refine(verified_text)
        return final_text

if __name__ == "__main__":
    # 模拟用户输入
    user_input = "我想了解一下深度学习中的损失函数创新有哪些？"
    # 模拟意图分类器
    intent_classifier = MockIntentClassifier()
    # 创建系统
    system = LiteratureReviewSystem(intent_classifier)
    # 生成综述
    review_text = system.generate_review(user_input)
    print(review_text)