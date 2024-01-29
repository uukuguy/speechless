help:
	@echo "Usage: make [lm_eval]"

		# --model_args model=speechless-code-mistral-7b-v1.0:Q4_K_M \
		# --model_args model=zephyr-7b-beta:Q4_K_M \
		# --model_args model=functionary-small-v2.2:Q4_K_M \


lm_eval_litellm:
	PYTHONPATH=$PWD \
	python -m lm_eval \
		--model openai-chat-completions \
		--model_args model=speechless-zephyr-code-functionary-7b:Q4_K_M \
		--tasks gsm8k \
		--gen_kwargs temperature=0.2,top_p=0.75 \
		--batch_size auto \
		--output_path ./eval_output
