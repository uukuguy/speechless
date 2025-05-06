
help:
        @echo "Usage: make [target]"
        @echo "Targets:"
        @echo "  preprocess_data: Preprocess the data"
        @echo "  train: Train the general reasoning model"

preprocess_data:
        python data_preprocess.py

#         --node-ip-address 28.160.2.28 \
#         --port 6379

ray_start:
        ray stop
        # ray start --head
        ray start --head \
          --dashboard-host 0.0.0.0 \
          --dashboard-port 8265 \

test_ray:
        python test_ray.py

HOSTNAME=$(shell hostname)
test_ray_remote:
        ray job submit --address http://${HOSTNAME}:8265 --working-dir . -- python test_ray_remote.py

train:
        bash train_general_reasoner.sh
