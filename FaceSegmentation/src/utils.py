import logging
import os

HINT_LEVEL = 5
logging.addLevelName(HINT_LEVEL, "HINT")


def hint(self: logging.Logger, message: str, *args: tuple, **kws: dict) -> None:
    """
    :Description:
    Function {hint} logs a message with level 'HINT' on this logger.

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
    :Description:
    Function {colored_log} logs a message with color based on the logging level.

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
    if level == 'ERROR':
        message = f"\n{'-' * (len(message))}\n{message}\n{'-' * (len(message))}\n\n"
    if level == 'CRITICAL':
        message = f"\n{'*' * (len(message) + 4)}\n* {message.upper()} *\n{'*' * (len(message) + 4)}\n\n"

    logger.log(getattr(logging, level, HINT_LEVEL if level == 'HINT' else None), f"{color}{message}{RESET}")


def set_paths() -> tuple[str, str, str, str, str]:
    """
    :Description:
    Function {set_paths} sets and returns the key directory paths for the project.

    :rtype: tuple[str, str, str, str, str]
    :return: A tuple containing paths for CURR_DIR, MAIN_DIR, WORK_DIR, IMGS_DIR, OUTPUT_COCO_DIR, OUTPUT_YOLO_DIR.
    """
    CURR_DIR = os.path.abspath(os.path.dirname(__file__))

    MAIN_EXT_DIR = os.path.abspath(os.path.join(CURR_DIR, '..', '..'))
    WORKING_DIR = os.path.join(MAIN_EXT_DIR, "work")

    IMAGES_DIR = os.path.join(MAIN_EXT_DIR, "constant/Assets/TestImages")
    OUTPUT_COCO_DIR = os.path.join(WORKING_DIR, "docks/Results/COCO")
    OUTPUT_YOLO_DIR = os.path.join(WORKING_DIR, "docks/Results/YOLO")

    for directory in [WORKING_DIR, IMAGES_DIR, OUTPUT_COCO_DIR, OUTPUT_YOLO_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    return MAIN_EXT_DIR, WORKING_DIR, IMAGES_DIR, OUTPUT_COCO_DIR, OUTPUT_YOLO_DIR


MAIN_DIR, WORK_DIR, IMGS_DIR, COCO_DIR, YOLO_DIR = set_paths()
