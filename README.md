# vision-compare

## Requirements

- `CUDA 10` (only for evaluation with GPU)
- `Python 3.7` (developed using `Python 3.7.6`)

## Setup

1. Create a Python virtual environment (e.g. `conda create -name tensorflow python=3.7.6` or `virtualenv env`)
2. Activate the environment (e.g. `conda activate tensorflow` or `source ./env/bin/activate`)
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `lib` directory and change into it
5. Clone the following repos with `git clone`
   - https://github.com/fizyr/keras-retinanet
   - https://github.com/experiencor/keras-yolo3
   - https://github.com/omni-us/squeezedet-keras
   - https://github.com/tanakataiki/ssd_kerasV2
6. Change the hyphens (`-`) in the directory names to underscores (`_`)
7. Change back to the root of the project directory
8. Install dependencies with `pip install -r requirements-dev.txt`, or if you are deploying on e.g. a Raspberry Pi `pip install -r requirements.txt`
9. Create a `model_data` directory and download the following models to it:
   - RetinaNet: https://github.com/fizyr/keras-retinanet/releases/tag/0.5.1
   - MobileNetv2 + SSD: https://github.com/tanakataiki/ssd_kerasV2
   - SSDv2 TFLite: https://coral.ai/models/
10. Rename the models as they are required in the scripts (more info will be added)
11. For development, clone the Python typings to the root of the project: https://github.com/python/typeshed
12. Download the COCO 2017 Train/Val annotations from https://cocodataset.org/#download to `data/COCO/annotations`
13. Download evaluation images and their annotations from the COCO dataset using `python src/download-coco.py`
14. Run the scripts from the root of the project directory (e.g. `python src/benchmark.py`)
