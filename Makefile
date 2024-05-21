all: requirements dataset 
	echo PATH="/home/gabrielalmeida/.local/bin:$PATH" > ~/.bashrc
	echo PJRT_DEVICE=TPU > ~/.bashrc

venv:
	sudo apt install python3.8-dev python3.8-venv
	python3 -m venv venv

requirements: requirements.txt
	pip3 install -r requirements.txt

dataset:
	mkdir dataset
	gcloud storage cp -r gs://watermark-detection-bucket/watermark/data/* dataset/

