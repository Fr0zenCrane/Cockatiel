# #!/usr/bin/env bash
conda create -n cockatiel python==3.10
conda activate cockatiel
pip install -e .
pip install transformers==4.37.2
pip install torch==2.1.2 torchvision==0.16.2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
