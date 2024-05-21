all: venv requirements dataset 
	export PATH="/home/gabrielalmeida/.local/bin:$PATH"
	export PJRT_DEVICE=TPU

venv:
	sudo apt install python3.8-dev python3.8-venv
	python3 -m venv venv

requirements: venv requirements.txt
	. venv/bin/activate; pip3 install torch~=2.3.0 torch_xla[tpu]~=2.3.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
	. venv/bin/activate; pip3 install -r requirements.txt

dataset:
	mkdir dataset
	gcloud storage cp -r gs://watermark-detection-bucket/watermark/data/* dataset/

