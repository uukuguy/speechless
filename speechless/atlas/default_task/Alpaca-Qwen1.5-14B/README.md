# Alpaca-Qwen1.5-14B on Atlas 800T A2

## Environment

### Install MindSpore

```bash
conda create -n env-mindspore python=3.9
conda activate env-mindspore

# for MindTransformers r1.0
export MS_VERSION=2.2.13
# for MindTransformers r1.10
# export MS_VERSION=2.3.0rc2

# aarch64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/aarch64/mindspore-${MS_VERSION/-/}-cp39-cp39-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install sympy
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl

python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
```
