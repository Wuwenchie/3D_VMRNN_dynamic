# 3D_VMRNN_dynamic
# Overview  
We propose model architecture based on a modified 3D-Geoformer with an encoder–decoder structure, integrating VMRNN, residual correction, and dynamic activation system. The input is 12 consecutive months of multivariable climate fields (τₓ, τᵧ wind stress and seven-layer ocean temperature anomalies, 9 channels total). The encoder extracts spatial features and converts them into sequences for VMRNN to capture long-term dependencies, while the decoder outputs 20-month predictions. The right panel shows the climate-aware dynamic activation system, which adaptively selects between ReLU and Learned Snake functions depending on ENSO states.  
![](https://github.com/Wuwenchie/3D_VMRNN_dynamic/blob/main/figure1.png)  
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
