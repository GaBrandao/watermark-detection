#!/usr/bin/env bash

export PATH="/home/gabrielalmeida/.local/bin:$PATH"
export PJRT_DEVICE=TPU

pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
pip install -r requirements.txt

mkdir dataset
gcloud storage cp -r gs://watermark-detection-bucket/watermark/data/* dataset/