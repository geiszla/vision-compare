import os
import sys
import warnings


# Ignore tnesorflow user/future warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set tensorflow log level to only log errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import tensorflow after the environment is set up
# pylint: disable=wrong-import-position
import tensorflow


def initialize_environment() -> None:
    """Initializes enviroment for the scripts

    Initializes the environment to be able to run the scripts in /lib directory properly and to
    make sure all necessary environment variables are set and required paths are added to
    sys.path.

    This method needs to be run before any /lib imports (this also means that lib imports cannot
    be done in the global scope, only in the methods they are used in, so that initialization is
    done before they are imported).
    """

    # Set other options to disable tensorflow logs
    tensorflow.get_logger().setLevel('ERROR')
    tensorflow.autograph.set_verbosity(0)
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

    # Add project root to path
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_path)

    # Add /lib directory and all directories inside to path
    lib_path = os.path.join(project_path, "lib")
    for directory_name in os.listdir(lib_path):
        sys.path.insert(0, os.path.join(lib_path, directory_name))
