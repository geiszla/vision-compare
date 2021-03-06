# Vision Compare

Vision Compare is a benchmark suite for object detection models. It provides a way for researchers to test detectors on the same data, same metrics and using the same hardware. This makes it possible to avoid comparing models by their reported
scores, which can have significant differences based on the test setup.

The benchmark takes advantage of Python class inheritance to build an abstraction of object detectors. This makes it easier to add more models to it as a form of a pluggable module. Any detector can be implemented for it in a way that it uses the `Detector` as their superclass and therefore standard operations can be performed on all of them without the need to directly use their implementation-specific API. This makes the benchmark very robust.

## Requirements

- `Python 3.7` (developed using `Python 3.7.7`)
- `CUDA 10.0` and `cuDNN 7.6` (only if you plan to use your GPU)

## Setup

1. Clone project with all its submodules (`git clone https://github.com/geiszla/vision-compare.git --recurse-submodules`)
2. Create a Python virtual environment (e.g. `conda create -n vision-compare python=3.7.7` or `virtualenv env`)
3. Activate the environment (e.g. `conda activate vision-compare` or `source ./env/bin/activate`)
4. Change into the project directory
5. Install required dependencies
   - Using [Poetry](https://github.com/python-poetry/poetry) (recommended)
      - Deployment: `poetry install --no-dev && pip install tensorflow==1.14.0`
      - Development: `poetry install`
   - Using Pip (only for deployment; can result in errors)
      - Deploying on Raspberry Pi: `pip install -r requirements-pi.txt && pip install tensorflow==1.14.0`
      - Deploying elsewhere: `pip install -r requirements.txt`
6. If you want to use a USB AI accelerator
   1. Install the [Edge TPU runtime](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime)
   2. install `tflite_runtime`
      - Raspberry Pi: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl`
      - Linux: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl`
      - Windows: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-win_amd64.whl`
      - MacOS: `pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl`
7. Create a `model_data` directory and place the weight files for the desired models there. For the default models, you can download them here:
   - [YOLOv3 320 and YOLOV3 tiny](https://pjreddie.com/darknet/yolo/) (convert to Keras model using `python lib/keras_yolo3_2/convert.py model_data/yolov3[-tiny].cfg model_data/yolov3[-tiny].weights model_data/yolov3[-tiny].h5`)
   - [ResNet 50](https://github.com/fizyr/keras-retinanet/releases) (rename to `retinanet.h5`)
   - [MobileNet v2 Lite for SSD 300](https://github.com/tanakataiki/ssd_kerasV2) (rename to `ssd.hdf5`)
   - [MobileNet v1 SSD](https://www.tensorflow.org/lite/models/object_detection/overview) (rename to `ssdv1.tflite`)
   - [MobileNet v1 and v2 SSD for Edge TPU](https://coral.ai/models/) (rename to `ssdv1_edgetpu.tflite` and `ssdv2_edgetpu.tflite` respectively)
   - SqueezeDet: you don't need to download this, as it comes with the repo

### Download image data

### VOC

1. Download VOC training/validation data from [their website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)
2. Extract `Annotations` and `JPEGImages` directories into the project's `data` directory
3. [Run the benchmark script](#running-the-scripts)

### COCO

Note that most of the default models are trained on COCO, so validation on it is redundant. If you still want to use the dataset, you need to modify the `data_generator` in `models_/detector.py` to load it instead of the VOC samples (you can also use `read_coco_annotations` function inside `utilities.py` to read downloaded data to the correct format).

1. Install `pycocotools` using `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`
2. Download the COCO 2017 Train/Val annotations from [their website](https://cocodataset.org/#download) and place it into `data/COCO/annotations` (create directory if doesn't exist)
3. Run `python src/download_coco.py` to download evaluation images and their annotations from the COCO dataset (by default, only 500 images and their annotations are downloaded; you can change this by modifying `IMAGE_COUNT` in the script)

### Install required packages on Linux

If you are deploying this project on Linux (especially the Raspberry Pi), you may be required to install a few additional packages as well:

```bash
sudo apt install libatlas-base-dev libjasper-dev libqtgui4 python3-pyqt5 libqt4-test libhdf5-dev
```

### Running the scripts

1. Activate the environment (e.g. `conda activate tensorflow` or `source ./env/bin/activate`; see instructions for creating an environment and downloading dependencies above)
2. Run the scripts from the root of the project directory (e.g. `python src/benchmark.py`)

## Project structure

Will be added
