


git clone https://gitee.com/ascend/MindSpeed-LLM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git

cd Megatron-LM
git checkout core_r0.7.0
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
mkdir logs model_from_hf model_weights dataset ckpt
cd ..


conda create -n env-mindspeed python=3.8
conda activate env-mindspeed
pip install torch==2.1.0 torch-npu==2.1.0
# pip install torchvision==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


git clone -b master https://gitee.com/ascend/apex.git
cd apex/
# dnf install patch # 安装patch
bash scripts/build.sh --python=3.8
cd apex/dist/
pip3 uninstall apex
pip3 install apex-0.1+ascend-cp38-cp38-linux_aarch64.whl
cd ../../..

# source ascend-toolkit 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# source atb库 环境变量
# Ascend nnal https://www.hiascend.ru/developer/download/community/result?module=pt&version=6.0.1.alpha001
source /usr/local/Ascend/nnal/atb/set_env.sh 



# 安装加速库
cd MindSpeed-LLM
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.7.0 in 2024.12.13
git checkout 4045864e6df
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -e .
cd ../..

## 安装其余依赖库
cd MindSpeed-LLM
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple