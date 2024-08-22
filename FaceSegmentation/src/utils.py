import logging
import os
import sys

HINT_LEVEL = 5
logging.addLevelName(HINT_LEVEL, "HINT")


def hint(self: logging.Logger, message: str, *args: tuple, **kws: dict) -> None:
    """
    Logs a message with level 'HINT' on this logger.

    :param self: The logger instance that this method is called on.
    :param message: The message to be logged.
    :param args: Additional positional arguments to be passed to the logger.
    :param kws: Additional keyword arguments to be passed to the logger.
    :rtype: None
    """
    if self.isEnabledFor(HINT_LEVEL):
        self._log(HINT_LEVEL, message, args, **kws)


logging.Logger.hint = hint
logging.basicConfig(level=HINT_LEVEL)
logger = logging.getLogger(__name__)

RESET = "\033[0m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD_BRIGHT_RED = "\033[1;91m"


def colored_log(level: str, message: str) -> None:
    """
    Logs a message with color based on the logging level.

    :param level: The logging level as a string ('HINT', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    :param message: The message to be logged.
    :rtype: None
    """
    color = {
        'HINT': BLUE,
        'INFO': GREEN,
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': BOLD_BRIGHT_RED,
    }.get(level, RESET)
    if level == 'CRITICAL':
        message = f"\n{'*' * 50}\n{message.upper()}\n{'*' * 50}\n"

    logger.log(getattr(logging, level, HINT_LEVEL if level == 'HINT' else None), f"{color}{message}{RESET}")


# def get_main_dir():
#     if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
#         print("Running in Google Colab")
#         return '/content'
#     try:
#         if 'ipykernel' in sys.modules:
#             print("Running in Jupyter Notebook or JupyterLab")
#             return os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
#     except ImportError:
#         pass
#     try:
#         print("Running locally")
#         return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
#     except NameError:
#         return os.path.abspath(os.getcwd())
#
#
# MAIN_DIR = get_main_dir()


def find_project_root(starting_dir, marker_file='setup.py'):
    current_dir = starting_dir
    while True:
        if os.path.exists(os.path.join(current_dir, marker_file)):
            return current_dir
        new_dir = os.path.dirname(current_dir)
        if new_dir == current_dir:
            return None
        current_dir = new_dir


def get_main_dir():
    try:
        starting_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        starting_dir = os.getcwd()

    project_root = find_project_root(starting_dir)

    if project_root is not None:
        return project_root
    else:
        return starting_dir


MAIN_DIR = get_main_dir()

# MAIN_DIR = os.getcwd()
IMGS_DIR = os.path.join(MAIN_DIR, "docks/TestImages")
COCO_DIR = os.path.join(MAIN_DIR, "docks/Results/COCO")
YOLO_DIR = os.path.join(MAIN_DIR, "docks/Results/YOLO")
