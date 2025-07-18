# 开放原子开源大赛 - 大模型科研工具创新赛

[官网地址](https://competition.atomgit.com/competitionInfo?id=0839e5d2820a02ee4616192432f04dac)

SpeechlessAI 团队参赛作品

- [](https://huggingface.co/uukuguy)
- [](https://github.com/uukuguy/speechless)

提交要求：文件名为 main.py ，运行时接受一个参数：--topic. 也就是说，运行方式为 python main.py --topic xxxx 。程序运行完毕后，生成综述到 review.md.

```text
.
├── 1.md # Build by `make topic_1` args: --topic "损失函数" --lang english 
├── 2.md # Build by `make topic_2` args: --topic "Text2SQL研究现状如何，面临哪些挑战？" --lang english 
├── 3.md # Build by `make topic_3` args: --topic "有哪些方法可以提升大模型的规划能力，各自优劣是什么？" --lang english 
├── 4.md # Build by `make topic_4` args: --topic "多模态大模型的技术发展路线是什么样的？"  --lang english 
├── Makefile
├── README.md
├── citation_utils.py # Generate citation
├── common_utils.py 
├── intent_recognition.py # Intent recognition
├── knowledge_base.py # Knowledge base
├── llm_utils.py # LLM utils
├── main.py # Main script
├── outputs
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Default support ZhipuAI. You can set OPENAI_BASE_URL and OPENAI_DEFAULT_MODEL to use other LLM.
export OPENAI_API_KEY="your zhipuai key" 

# The script execution time is relatively long, and the intermediate results of multiple steps are cached under the path outputs/{topic}_{lang}. 
# If the script execution is interrupted midway, you can re-execute the script to automatically load the completed cache and continue execution. 
# If a complete re-execution is needed, you must delete the corresponding cache directory.

python main.py --topic "your topic" --output_file "default is review.md" --lang "default is english, support chinese"

```
