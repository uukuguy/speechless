# Ascend SFT

## Setup Developing Environment

### 0. Conda Environment (env-ascend)

```bash
# conda create -n env-mindspeed python=3.8
# conda activate env-mindspeed

conda create -n env-ascend python=3.10
conda activate env-ascend
export ASCEND-SFT=$HOME/sandbox/LLM/speechless.ai/tasks/ascend-sft
```

### 1. Install PyTorch

```bash
# pip install torch==2.1.0 torch-npu==2.1.0 torchvision==0.16.0
pip install torch==2.4.0 torch-npu==2.4.0.post2 torchvision==0.19.0
```

### 2. Install Apex

```bash
git clone -b master https://gitee.com/ascend/apex.git
cd apex/
bash scripts/build.sh --python=3.8
cd apex/dist/
pip3 uninstall apex
pip3 install apex-0.1+ascend-cp38-cp38-linux_aarch64.whl
cd ../../..
```

### 3. Install MindSpeed-LLM

```bash
git clone https://gitee.com/ascend/MindSpeed-LLM.git
git clone https://github.com/NVIDIA/Megatron-LM.git

cd Megatron-LM
git checkout core_r0.7.0
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
mkdir logs model_from_hf model_weights dataset ckpt
cd ..
```

### Setup Ascend Environment

```bash
# source atb库 环境变量
# Ascend nnal https://www.hiascend.ru/developer/download/community/result?module=pt&version=6.0.1.alpha001
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh 
```

### TorchAir

```bash
mkdir TorchAir
cd TorchAir
git clone https://gitee.com/ascend/torchair
cd torchair
git submodule update --init --recursive
bash ./configure
# ASCEND_SDK_PATH=/usr/local/Ascend/ascend-toolkit/latest

mkdir build
cd build
cmake ..
make torchair -j8
make install_torchair
# source TorchAir/torchair/tools/env.sh
```

### 安装加速库 MindSpeed

```bash
cd MindSpeed-LLM
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.7.0 in 2024.12.13
<!-- git checkout 4045864e6df -->
# git checkout 1.0.RC3_core_r0.7.0
git checkout core_r0.7.0
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -v -e .
cd ../..
```

### 安装其余依赖库

```bash
cd MindSpeed-LLM
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

bash convert_hf_to_mcore_llama_2_7b.sh

```bash
#!/bin/bash

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 4 \
    --num-layer-list 8,8,8,8 \
    --model-type-hf llama2 \    --use-mcore-models \
    --load-dir ./model_from_hf/Llama-2-7b-hf/ \
    --save-dir ./model_weights/Llama-2-7b-mcore/ \
    --tokenizer-model ./model_from_hf/llama-2-7b-hf/tokenizer.model
```

[MindSpeed-LLM 单样本指令微调](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/features/instruction_finetune.md)

- Alpca/ShareGPT 风格数据
- 支持QLoRA

[MindSpeed-LLM 多轮对话微调](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/features/multi-turn_conversation.md)

Alpaca风格数据

[多样本Pack微调](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/features/multi-sample_pack_fine-tuning.md)



compiling dataset index builder ...          
make: relocation error: /usr/lib64/libguile-2.0.so.22: symbol ffi_type_uint32 version LIBFFI_BASE_7.0 not defined in file libffi.so.7 with link time reference
ERROR:megatron.core.datasets.utils:Failed to compile the C++ dataset helper functions

https://gitee.com/ascend/MindSpeed-LLM/issues/I9G2AM

1.在ModelLink/megatron/core/datasets目录下手动make编译
2.注释掉ModelLink/modellink/initialize.py中的#compile_helpers()方法

./MindSpeed/mindspeed/mindspore/training/initialize.py: compile_helpers()
MindSpeed/mindspeed/initialize.py compile_helpers()
./megatron-core_r0.7.0/training/initialize.py:        compile_helpers() 
./mindspeed_llm/training/initialize.py:        compile_helpers() 




Llama-2-7B
MMLU:

                                subject  question_n       acc         
0                        moral_disputes         346  0.517341
1                 professional_medicine         272  0.522059               
2                            prehistory         324  0.484568
3                     international_law         121  0.603306
4                             astronomy         152  0.434211                                  
5                electrical_engineering         145  0.496552                                  
6                       human_sexuality         131  0.541985
7          high_school_european_history         165  0.600000                                  
8                 high_school_chemistry         203  0.369458                                  
9                         miscellaneous         783  0.630907                                  
10              professional_psychology         612  0.449346                                  
11                    logical_fallacies         163  0.490798                                  
12               elementary_mathematics         378  0.269841           
13           high_school_macroeconomics         390  0.451282                                  
14                   conceptual_physics         235  0.434043                                  
15                        jurisprudence         108  0.537037                                  
16                     abstract_algebra         100  0.300000
17             college_computer_science         100  0.350000                                  
18                            nutrition         306  0.493464                                  
19                     security_studies         245  0.538776            
20               high_school_statistics         216  0.259259
21                     machine_learning         112  0.375000
22                    college_chemistry         100  0.330000
23                      college_biology         144  0.451389                                  
24                              anatomy         135  0.481481                                  
25              professional_accounting         282  0.351064  
26                     public_relations         110  0.527273
27               high_school_us_history         204  0.519608                                  
28                  high_school_physics         151  0.337748
29            high_school_world_history         237  0.654008                    
30                      college_physics         102  0.196078
31                           management         103  0.514563
32                  high_school_biology         310  0.509677
33                     college_medicine         173  0.416185
34                          human_aging         223  0.556054
35                            sociology         201  0.621891
36                             virology         166  0.409639
37  high_school_government_and_politics         193  0.652850
38                         formal_logic         126  0.293651
39                     professional_law        1534  0.370926
40                    computer_security         100  0.610000
41                      business_ethics         100  0.500000
42               high_school_psychology         545  0.603670
43                      world_religions         171  0.684211
44                           philosophy         311  0.588424
45                      moral_scenarios         895  0.246927
46                    us_foreign_policy         100  0.660000
47              high_school_mathematics         270  0.285185
48           high_school_microeconomics         238  0.424370
49                high_school_geography         198  0.479798
50                         econometrics         114  0.289474
51                            marketing         234  0.670940
52                     medical_genetics         100  0.510000
53         high_school_computer_science         100  0.390000
54                  college_mathematics         100  0.300000
55                         global_facts         100  0.330000
56                   clinical_knowledge         265  0.456604
57                                total       14042  0.456844
total: 100%|███████████████████████████████████████████████████| 57/57 [45:04<00:00, 47.46s/it]
INFO:__main__:MMLU Running Time:, 2704.9518427848816

alpaca: 14042: 
2000 steps: 0.388620

speechless-thoughts-252k
10000 steps: 0.437972
8000 steps:  0.435764
6000 steps:  0.426079
4000 steps:  0.434269
2000 steps:  0.430922

Infinity-Instruct-50K
2000 steps: 0.398376

Infinity-Instruct-250K
750 steps: 0.417676

Infinity-Instruct-1M
3000 steps: 0.437473

Infinity-Instruct-3M
9000 steps: 0.441746

Infinity-Instruct-9M
18000 steps: 0.445283
27000 steps: 0.429

TPxPP
1x1 out of memory
1x4 out of memory
1x8 6500 ms/it
2x2 7500 ms/it
4x2 7500 ms/it
2x4 7200 ms/it
8x1 9540 ms/it
