all: env requirements dataset 
	export PATH="/home/gabrielalmeida/.local/bin:$PATH"
	export PJRT_DEVICE=TPU

requirements:
	pip3 install torch~=2.3.0 torch_xla[tpu]~=2.3.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
	pip3 install -r requirements.txt

env:
	python3 -m venv env
	source env/bin/activate

dataset:
	mkdir dataset
	gcloud storage cp -r gs://watermark-detection-bucket/watermark/data/* dataset/

