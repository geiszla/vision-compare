# Vision Compare

## Requirements

- `CUDA 10` (only for GPU compatibility)
- `Python 3.7` (developed using `Python 3.7.7`)

## Setup

1. Clone project with all its submodules (`git clone https://github.com/geiszla/vision-compare.git --recurse-submodules`)
2. Create a Python virtual environment (e.g. `conda create -n vision-compare python=3.7.7` or `virtualenv env`)
3. Activate the environment (e.g. `conda activate vision-compare` or `source ./env/bin/activate`)
4. Change into the project directory
5. Install required dependencies
   - Using [Poetry](https://github.com/python-poetry/poetry) (recommended)
      - Deployment: `poetry install --no-dev`
      - Development: `poetry install`
   - Using Pip (only for deployment; can result in errors)
      - Deploying on Raspberry Pi: `pip install -r requirements-pi.txt`
      - Deploying elsewhere: `pip install -r requirements.txt`
6. Install optional dependencies:
   - If you want to use the COCO image downloader script, install `pycocotools` using `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`
   - If you want to use a USB AI accelerator, install `tflite_runtime`
      - Raspberry Pi: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl`
      - Linux: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl`
      - Windows: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-win_amd64.whl`
      - MacOS: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl`
7. Create a `model_data` directory and place the weight files for the desired models there. For the default models, you can download them here (prefer the models trained on COCO for better results):
   - [YOLOv3 320](https://pjreddie.com/darknet/yolo/)
   - [YOLOv3 320 config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - [ResNet 50](https://github.com/fizyr/keras-retinanet/releases/tag/0.5.1) (rename to `retinanet.h5`)
   - [MobileNet v2 Lite for SSD 300](https://github.com/tanakataiki/ssd_kerasV2) (rename to `ssd.hdf5`)
   - [MobileNet v1 SSD](https://coral.ai/models/) (rename to `ssdv1_edgetpu.tflite`)
   - [MobileNet v2 SSD](https://coral.ai/models/) (rename to `ssdv2_edgetpu.tflite`)
   - SqueezeDet: you don't need to download this, as it comes with the repo

### Download image data

See the instructions above for installing dependencies for the download script

1. Download the COCO 2017 Train/Val annotations from [their website](https://cocodataset.org/#download) and place it into `data/COCO/annotations` (create directory if doesn't exist)
2. Download evaluation images and their annotations from the COCO dataset using `python src/download_coco.py`

### Install required packages on Linux

If you are deploying this project on the Raspberry Pi, you may be required to install a few additional packages as well:

```bash
sudo apt install libatlas-base-dev libjasper-dev libqtgui4 python3-pyqt5 libqt4-test libhdf5-dev
```

### Running the scripts

1. Activate the environment (e.g. `conda activate tensorflow` or `source ./env/bin/activate`; see instructions for creating an environment and downloading dependencies above)
2. Run the scripts from the root of the project directory (e.g. `python src/benchmark.py`)

## Project structure

Will be added
