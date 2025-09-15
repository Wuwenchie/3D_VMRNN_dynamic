# 3D_VMRNN_dynamic
# Installation  
    conda env create -f environment.yml
    conda activate 3D_VMRNN
    pip install -e .
    pip install einops
    pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install packaging timm==0.6.11 pytest chardet yacs termcolor submitit tensorboardX triton==2.0.0 fvcore
    pip install mlflow
    pip install causal_conv1d==1.1.1
    pip install mamba_ssm==1.1.1  
# Training
    cd code
    python trainer.py
# Testing
    cd code
    python test_model.py
