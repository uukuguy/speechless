help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  preprocess_data: Preprocess the data"
	@echo "  train: Train the general reasoning model"

preprocess_data:
	python data_preprocess.py

#	--node-ip-address 28.160.2.28 \
#	--port 6379

# Calculate the number of GPUs based on the CUDA_VISIBLE_DEVICES variable. Use all the GPUs if not set.
ifneq ($(CUDA_VISIBLE_DEVICES),)
        GPU_COUNT := $(shell echo $(CUDA_VISIBLE_DEVICES) | tr ',' '\n' | wc -l)
else
        GPU_COUNT := $(shell nvidia-smi -L | wc -l)
endif

ray_start:
	ray stop
	# ray start --head
	ray start --head \
		--disable-usage-stats \
		--num-gpus ${GPU_COUNT} \
		--dashboard-host 0.0.0.0 \
		--dashboard-port 8265 \

test_ray:
	python test_ray.py

HOSTNAME=$(shell hostname)
test_ray_remote:
	ray job submit --address http://${HOSTNAME}:8265 --working-dir . -- python test_ray_remote.py

train:
	bash train_general_reasoner.sh
