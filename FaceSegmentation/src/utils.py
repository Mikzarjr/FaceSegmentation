import inspect
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

colors = {
    "RESET": "30",
    "RED": "31",
    "GREEN": "32",
    "YELLOW": "33",
    "BLUE": "34",
    "MAGENTA": "35",
    "CYAN": "36",
    "WHITE": "37",
    "BRIGHT_BLACK": "1;90",
    "BRIGHT_RED": "1;91",
    "BRIGHT_GREEN": "1;92",
    "BRIGHT_YELLOW": "1;93",
    "BRIGHT_BLUE": "1;94",
    "BRIGHT_MAGENTA": "1;95",
    "BRIGHT_CYAN": "1;96",
    "BRIGHT_WHITE": "1;97"
}


def colored_log(level: str, message: str) -> None:
    """
    :Description:
    Function {colored_log} logs a message with color based on the logging level.

    :param level: The logging level as a string ('HINT', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    :param message: The message to be logged.
    :rtype: None
    """
    color = {
        'HINT': "BLUE",
        'INFO': "GREEN",
        'WARNING': "YELLOW",
        'ERROR': "RED",
        'CRITICAL': "BOLD_BRIGHT_RED",
    }.get(level)
    if level == 'ERROR':
        message = (f":\t{message}\n"
                   f"{'-' * (max(len(message.split('\n')[0]) + 20, len(message.split('\n')[-1])))}"
                   f"\n\n")
    if level == 'CRITICAL':
        message = f"\n{'*' * (len(message) + 4)}\n* {message.upper()} *\n{'*' * (len(message) + 4)}\n\n"

    return logger.log(getattr(logging, level, HINT_LEVEL if level == 'HINT' else None), colored_string(message, color))


def get_error_origin():
    frame = inspect.currentframe()
    caller_frame = frame.f_back.f_back
    filename = caller_frame.f_code.co_filename
    lineno = caller_frame.f_lineno
    funcname = caller_frame.f_code.co_name
    return frame, caller_frame, filename, lineno, funcname


def colored_string(string, color):
    cc = colors.get(color.upper())
    if cc:
        begin = f"\033[{cc}m"
        end = "\033[0m"
        return begin + string + end
    else:
        frame, caller_frame, filename, lineno, funcname = get_error_origin()

        error_message = (f"Unproper color: '{color}' in string: '{string}'\n"
                         f"In file '{filename}', function '{funcname}', line {lineno}")

        colored_log("ERROR", error_message)
        return ''


print(colored_string("This sentence should be green", "bright_green"))
print(colored_string("This sentence should be green", "bright green"))


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
