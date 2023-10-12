
create_docker:
	bash create_docker.sh

run_docker:
	docker start postgres-sql-eval

stop_docker:
	docker stop postgres-sql-eval

init_db:
	./data/init_db.sh academic advising atis geography restaurants scholar yelp

# OPENAI_MODEL=gpt-3.5-turbo-0613
OPENAI_MODEL=gpt-4-0613
# OPENAI_MODEL=phind-codellama-34b-v2
#OPENAI_MODEL=phind-codellama-34b-v2-gptq

# openai prompt
# Correct: 86/175(49.14%), 1:24:30, 28.97s/bit
# Average rate of exact match: 44%
# Average correct rate: 49.14%
# codellam prompt
# OPENAI_MODEL=speechless-codellama-34b-v1.0
# OPENAI_MODEL=sqlcoder_phind

# # openai prompt: FAILED
# # codellama prompt
# # Correct: 48/175(27.43%), 44:29, 15.25s/bit
# # Average rate of exact match: 25.14%
# # Average correct rate: 27.43%
# OPENAI_MODEL=speechless-codellama-airoboros-orca-platypus-13b

openai:
	mkdir -p results/${OPENAI_MODEL} # create directory for storing results
	python main.py \
	-q data/questions_gen.csv \
	-o results/${OPENAI_MODEL}/my_query_generator.csv \
	-g oa \
	-f prompts/phind_prompt.md \
	-m ${OPENAI_MODEL} \
	--verbose \
	--timeout_gen 180 \
	-n 175 \
	-p 1

# openai prompt
# Correct: 86/175(49.14%), 1:24:30, 28.97s/bit
# Average rate of exact match: 44%
# Average correct rate: 49.14%
#
# codellam prompt
# Correct so far: 84/175 (48.00%): 100%|█████████| 175/175 [1:00:50<00:00, 20.86s/it]
# Average rate of exact match: 40.00%
# Average correct rate: 48.00%
# CODELLAMA_MODEL=speechless-codellama-34b-v1.0

# # openai prompt: FAILED
# #
# # codellama prompt
# # Correct: 48/175(27.43%), 44:29, 15.25s/bit
# # Average rate of exact match: 25.14%
# # Average correct rate: 27.43%
# CODELLAMA_MODEL=speechless-codellama-airoboros-orca-platypus-13b

# CODELLAMA_MODEL=speechless-codellama-dolphin-orca-platypus-13b
#CODELLAMA_MODEL=uukuguy-speechless-codellama-dolphin-orca-platypus-13b
#CODELLAMA_MODEL=uukuguy-speechless-codellama-dolphin-orca-platypus-34b

# CODELLAMA_MODEL=sqlcoder_phind
# CODELLAMA_MODEL=speechless-codellama-airoboros-orca-platypus-13b-nl2sql
# CODELLAMA_MODEL=speechless-codellama-airoboros-orca-platypus-13b-codellama
# CODELLAMA_MODEL=speechless-codellama-airoboros-orca-platypus-13b
# CODELLAMA_MODEL=CodeLlama-13b-hf
#CODELLAMA_MODEL=Xwin-LM-70B-V0.1-GPTQ
# CODELLAMA_MODEL=Llama-2-7b-hf

# # ---------- NumbersStation/nsql-llama-2-7B ----------
# # nl2sql prompt:
# # Correct so far: 53/175 (30.29%): 100%|█████████████| 175/175 [01:20<00:00,  2.18it/s]
# # Average rate of exact match: 26.29%
# # Average correct rate: 30.29%
# CODELLAMA_MODEL=nsql-llama-2-7B


# # ---------- speechlessai/speechless-baichuan2-dolphin-orca-platypus-13b ----------
# # nl2sql prompt:
# # Correct so far: 56/175 (32.00%): 100%|████████████| 175/175 [03:39<00:00,  1.26s/it]
# # Average rate of exact match: 30.86%
# # Average correct rate: 32.00%
# CODELLAMA_MODEL=speechless-baichuan2-dolphin-orca-platypus-13b


# # ---------- speechlessai/speechless-codellama-airoboros-orca-platypus-13b ----------
# # nl2sql prompt:
# # Correct so far: 85/175 (48.57%): 100%|████████████| 175/175 [02:23<00:00,  1.22it/s]
# # Average rate of exact match: 44.57%
# # Average correct rate: 48.57%
# CODELLAMA_MODEL=speechless-codellama-airoboros-orca-platypus-13b


# # ---------- speechlessai/speechless-codellama-dolphin-orca-platypus-13b ----------
# # nl2sql prompt:
# # Correct so far: 93/175 (53.14%): 100%|████████████| 175/175 [02:21<00:00,  1.24it/s]
# # Average rate of exact match: 46.29%
# # Average correct rate: 53.14%

# CODELLAMA_MODEL=speechless-codellama-dolphin-orca-platypus-13b


# # ---------- dffog/sqlcoder ----------
# #  nl2sql prompt:
# #  Correct so far: 107/175 (61.14%): 100%|███████████| 175/175 [03:42<00:00,  1.27s/it]
# #  Average rate of exact match: 58.29%
# #  Average correct rate: 61.14%
# CODELLAMA_MODEL=sqlcoder


# # ---------- speechless-codellama-34b ----------
# # backend: vllm-gptq, async_generate(), nl2sql_prompt
# # Correct so far: 118/175 (67.43%): 100%|███████| 175/175 [05:02<00:00,  1.73s/it]
# # Average rate of exact match: 62.86%
# # Average correct rate: 67.43%

CODELLAMA_MODEL=speechless-codellama-34b


# CODELLAMA_MODEL=Phind-CodeLlama-34B-v2-GPTQ
# CODELLAMA_MODEL=Xwin-LM-70B-V0.1-GPTQ
# CODELLAMA_MODEL=CodeLlama-34B-Instruct-GPTQ


# # ---------- TheBloke/Phind-CodeLlama-34B-v2-GPTQ ----------
# # exllamav2, async_generate(), nl2sql_prompt 
# # Correct so far: 91/175 (52.00%): 100%|████████| 175/175 [19:46<00:00,  6.78s/it] 
# # Average rate of exact match: 48.00% 
# # Average correct rate: 52.00%
# CODELLAMA_MODEL=Phind-CodeLlama-34B-v2-GPTQ


# # ---------- TheBloke/CodeLlama-34B-Instruct-GPTQ ----------
# # exllamav2, async_generate(), nl2sql_prompt
# # Correct so far: 86/175 (49.14%): 100%|████████| 175/175 [11:45<00:00,  4.03s/it]
# # Average rate of exact match: 41.14%
# # Average correct rate: 49.14%
# CODELLAMA_MODEL=CodeLlama-34B-Instruct-GPTQ

# CODELLAMA_MODEL=Xwin-LM-70B-V0.1-GPTQ


codellama:
	mkdir -p results/${CODELLAMA_MODEL} # create directory for storing results
	python main.py \
	-q data/questions_gen.csv \
	-o results/${CODELLAMA_MODEL}/my_query_generator.csv \
	-g oa \
	-f prompts/speechless_nl2sql_prompt.md \
	-m ${CODELLAMA_MODEL} \
	--verbose \
	--timeout_gen 180 \
	--temperature 0.01 \
	--max_tokens 600 \
	--sampling_method beam_search \
	--num_beams 4 \
	--best_of 4 \
	--stop "\`\`\`" \
	-n 175 \
	-p 10



