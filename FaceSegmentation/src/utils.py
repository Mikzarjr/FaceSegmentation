import logging
import os

HINT_LEVEL = 5
logging.addLevelName(HINT_LEVEL, "HINT")


def hint(self: object, message: object, *args: object, **kws: object) -> object:
    """
    :Description:
    Function {hint} ...

    :param self:
    :param message:
    :param args:
    :param kws:
    :rtype: object
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


def colored_log(level: object, message: object) -> object:
    """
    :Description:
    Function {colored_log} ...


    :param level:
    :param message:
    :rtype: object
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


colored_log('HINT', "This is a hint message.")
colored_log('INFO', "This is an info message.")
colored_log('WARNING', "This is a warning message.")
colored_log('ERROR', "This is an error message.")
colored_log('CRITICAL', "This is a critical message.")

MAIN_DIR = os.getcwd()
IMGS_DIR = os.path.join(MAIN_DIR, "docks/TestImages")
COCO_DIR = os.path.join(MAIN_DIR, "docks/Results/COCO")
YOLO_DIR = os.path.join(MAIN_DIR, "docks/Results/YOLO")
