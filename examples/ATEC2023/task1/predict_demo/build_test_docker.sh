#!/usr/bin/bash

export IMG_ID=atec-857
export IMG_TAG=task1_test_`date +%Y%m%d-%H%M%S`
echo ${IMG_ID}
echo ${IMG_TAG}

export docker_url=atec-image-2022-annual-registry-vpc.cn-beijing.cr.aliyuncs.com/zark

make docker_token
make docker_login

docker build -f Dockerfile_test --network=host -t ${IMG_ID}:${IMG_TAG} . && \
docker tag ${IMG_ID}:${IMG_TAG} ${docker_url}/${IMG_ID}:${IMG_TAG} && \
docker push ${docker_url}/${IMG_ID}:${IMG_TAG}


${HOME}/adabench_cli submit \
	--dataset_id atec_2023_task_1_a  \
	--image ${docker_url}/${IMG_ID}:${IMG_TAG} \
	--solution_type shell

${HOME}/adabench_cli run -t image:///home/admin/atec_project:predict \
	--dataset atec_2023_task_1_a \
	--image ${docker_url}/${IMG_ID}:${IMG_TAG} \
	--solution_type shell

docker rmi ${docker_url}/${IMG_ID}:${IMG_TAG}
docker rmi ${IMG_ID}:${IMG_TAG}
