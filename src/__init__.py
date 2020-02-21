import sys
from os import listdir, path

PROJECT_PATH = path.abspath(path.join(path.dirname(__file__), ".."))

def initialize_environment():
    sys.path.append(PROJECT_PATH)

    lib_path = path.join(PROJECT_PATH, "lib")
    for directory_name in listdir(lib_path):
        if directory_name != 'deep_sort_yolov3':
            sys.path.append(path.join(lib_path, directory_name))
