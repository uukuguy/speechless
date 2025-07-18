
dist_train:
	CUDA_VISIBLE_DEVICES=0,1,2,3 \
	accelerate launch --num_processes 3 \
		--config_file ./deepspeed_zero3.yaml \
		run_r1_grpo.py \
		--config ./grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
