import logging
import os

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


MAIN_DIR = os.getcwd()
IMGS_DIR = os.path.join(MAIN_DIR, "docks/TestImages")
COCO_DIR = os.path.join(MAIN_DIR, "docks/Results/COCO")
YOLO_DIR = os.path.join(MAIN_DIR, "docks/Results/YOLO")
