docker_token:
	${HOME}/adabench_cli get-docker-token --dataset atec_2023_task_3_a

docker_login:
	docker login --username=cr_temp_user atec-image-2022-annual-registry-vpc.cn-beijing.cr.aliyuncs.com

run_id=e02601fc-9d90-11ee-bac9-00163e0e3cb2

run_desc:
	${HOME}/adabench_cli run-desc --run_id ${run_id}

run_log:
	${HOME}/adabench_cli run-log --run_id ${run_id}

download_model:
	${HOME}/adabench_cli run-download --run_id ${run_id} --download_type model