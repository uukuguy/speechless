#!/usr/bin/env python

import re
from rich.console import Console
from rich.markdown import Markdown
from loguru import logger
from llm_utils import LLMClient

console = Console()
# Usage: console.print(Markdown(generated_text))

def replace_with_html_entities(text):
    entities = {
        '[':'\[',
        ']':'\]',
        '_':'\_',
        '&': '&amp;',
        '©': '&copy;',
        '®': '&reg;',
        '™': '&trade;',
        '€': '&euro;',
        '£': '&pound;',
        '¥': '&yen;',
        '¢': '&cent;',
        '—': '&mdash;',
        '–': '&ndash;',
        '•': '&bull;',
        '…': '&hellip;'
    }
    for char, entity in entities.items():
        text = text.replace(char, entity)
    return text

def wrap_specific_tags_with_cdata(tag_name, content):
    tag_pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    replacement = f'<{tag_name}><![CDATA[\\1]]></{tag_name}>'
    return re.sub(tag_pattern, replacement, content, flags=re.DOTALL)

def do_generate_paragraph_of_review(query: str, TOPIC: str, llm_client):

    HEAD="""Based on the in-depth details extracted from the file related to '"""
    MIDDLE0=f"""', construct an analytical and comprehensive review section on {TOPIC}, emphasizing '"""
    MIDDLE1="""'.
    While developing the content, adhere to the following protocols:
    1. **Accurate Citations**: Reference specific content from the file by embedding the actual DOI numbers furnished, without any alterations. Utilize the format '[Placeholder_Of_DOI]' right after the sentence where the reference is applied. 
    2. **Strict Adherence**: Stick to the particulars and DOI details from the file; avoid integrating external or speculative data.
    3. **Scientific Language**: Uphold a technical and scholarly diction akin to chemical engineering literature.
    4. **Format & Translation**: After creating the main review content, append an 'integrative understanding and prospective outlook' section within the same <English></English> and <Chinese></Chinese> XML tags, demarcated with '※※※'. This segment should transcend a mere summation and foster a forward-thinking discussion, potentially elucidating future directions and broader horizons grounded in the file's content. 
    The content structure should resemble:
    <example>
            <English> 
                    Detailed analysis established from the study of reference [Placeholder_Of_DOI1]. Synthesized comprehension stemming from references [Placeholder_Of_DOI2] and [Placeholder_Of_DOI3]. 
                    ※※※
                    Integrative understanding and prospective outlook: Taking into consideration the advancements and findings discussed in the file, there lies an opportunity to explore emerging fields and innovative methodologies. Future research endeavors might focus on ...
            </English>
            <Chinese> 
                    基于[Placeholder_Of_DOI1]参考文献的深度分析。从[Placeholder_Of_DOI2]和[Placeholder_Of_DOI3]的参考文献中获得的综合理解。
                    ※※※
                    综合理解与未来展望: 考虑到文件中讨论的先进成果和发现，我们有机会探索新兴领域和创新方法。未来的研究努力可能会集中在...
            </Chinese>
            <References> 
                    Placeholder_Of_DOI1
                    Placeholder_Of_DOI2
                    Placeholder_Of_DOI3
            </References>
    </example>
    In the 'integrative understanding and prospective outlook' segment, aspire to:
    - **Offer an expansive perspective**: Illuminate potential pathways and pioneering research opportunities, grounded in the details divulged in the file.
    - **Propose forward-thinking suggestions**: Advocate for innovative angles and burgeoning domains that might take center stage in future explorations, while rooted in the file's details.
    Finally, compile all the cited DOIs in the 'References' compartment, adhering to the <References></References> XML tag, using the exact DOIs designated in the file.
    <file-attachment-contents filename="""
    END="""
    </file-attachment-contents>
    """

    # document = open(f'{i+1}/EnglishWithQuotes.txt', 'r', encoding='UTF8').read().strip()[:-1]
    prompt = replace_with_html_entities(HEAD + QuestionsForReview[i] + \
            MIDDLE0 + \
            ParagraphQuestionsForReview[i] + \
            MIDDLE1 + \
            f'"Paragraph{i+1}Info.txt">\n') + \
            document + \
            END


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--review_type", type=str, default="concept", choices=[r.value for r in ReviewType] help="综述类型")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    query = args.query

    model_name = os.getenv("OPENAI_DEFAULT_MODEL")
    llm_client = LLMClient(model_name=model_name)

    from intent_recognition import analysis_query
    TOPIC, review_type = analysis_query(query, llm_client, review_type_for_check=args.review_type)

    do_generate_paragraph_of_review(query, TOPIC)