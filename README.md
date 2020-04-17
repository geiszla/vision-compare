# vision-compare

## Requirements

- `CUDA 10` (only for evaluation with GPU)
- `Python 3.7` (developed using `Python 3.7.7`)

## Setup

1. Create a Python virtual environment (e.g. `conda create -n vision-compare python=3.7.7` or `virtualenv env`)
2. Activate the environment (e.g. `conda activate vision-compare` or `source ./env/bin/activate`)
3. Install required dependencies
   - Using [Poetry](https://github.com/python-poetry/poetry) (recommended)
      - Deployment: `poetry install --no-dev`
      - Development: `poetry install`
   - Using Pip (only for deployment; can result in errors)
      - Deploying on Raspberry Pi: `pip install -r requirements-pi.txt`
      - Deploying elsewhere: `pip install -r requirements.txt`
4. Install optional dependencies:
   - If you want to use a USB AI accelerator, install `tflite_runtime`
      - Raspberry Pi: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl`
      - Linux: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl`
      - Windows: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-win_amd64.whl`
      - MacOS: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl`
   - If you want to use the COCO image downloader script, install `pycocotools` using `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`
5. Create a `model_data` directory and download the following weight files into it:
   - RetinaNet: https://github.com/fizyr/keras-retinanet/releases/tag/0.5.1
   - MobileNetv2 + SSD: https://github.com/tanakataiki/ssd_kerasV2
   - SSDv2 TFLite: https://coral.ai/models/
6. Rename the models as they are required in the scripts (more info will be added)

### Download image data

See the instructions above for installing dependencies for the download script

1. Download the COCO 2017 Train/Val annotations from [their website](https://cocodataset.org/#download) and place it into `data/COCO/annotations`
2. Download evaluation images and their annotations from the COCO dataset using `python src/download_coco.py`

### Running the scripts

1. Activate the environment (e.g. `conda activate tensorflow` or `source ./env/bin/activate`; see instructions for creating an environment above)
2. Run the scripts from the root of the project directory (e.g. `python src/benchmark.py`)

## Project structure

Will be added
