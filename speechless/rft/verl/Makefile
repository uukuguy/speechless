help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  preprocess_data: Preprocess the data"
	@echo "  train: Train the general reasoning model"

preprocess_data:
	python data_preprocess.py \
		--dataset-name ../../difc2025_round_a_rft_train_data_all-8k.jsonl \
		--local-dir ./data/difc2025-round-a

# Calculate the number of GPUs based on the CUDA_VISIBLE_DEVICES variable. Use all the GPUs if not set.
ifneq ($(CUDA_VISIBLE_DEVICES),)
        NUM_GPUS := $(shell echo $(CUDA_VISIBLE_DEVICES) | tr ',' '\n' | wc -l)
else
        NUM_GPUS := $(shell nvidia-smi -L | wc -l)
endif

ray_start:
	ray stop
	# ray start --head
	ray start --head \
		--disable-usage-stats \
		--num-gpus ${NUM_GPUS} \
		--dashboard-host 0.0.0.0 \
		--dashboard-port 8265 \

test_ray:
	python test_ray.py

HOSTNAME=$(shell hostname)
test_ray_remote:
	ray job submit --address http://${HOSTNAME}:8265 --working-dir . -- python test_ray_remote.py

train:
	bash train_general_reasoner.sh
