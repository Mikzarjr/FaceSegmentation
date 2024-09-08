import inspect
import linecache
import logging
import os
import re
import sys
from typing import Optional

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


def colored_log(level: str, message: str) -> None:
    """
    :Description:
    Logs a message with color based on the logging level.

    :param level: The logging level as a string ('HINT', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    :param message: The message to be logged.
    :rtype: None
    """
    logging.Logger.hint = hint
    logging.basicConfig(level=HINT_LEVEL)
    logger = logging.getLogger(__name__)
    color = {
        'HINT': "BRIGHT_BLUE",
        'INFO': "BRIGHT_GREEN",
        'WARNING': "BRIGHT_YELLOW",
        'ERROR': "BRIGHT_RED",
        'CRITICAL': "BRIGHT_RED",
    }.get(level)
    if level == 'ERROR':
        message = (f":\t{message}\n"
                   f"{'-' * (max(visible_length(message.split('\n')[0]) + 20, visible_length(max(message.split('\n'), key=visible_length))))}"
                   f"\n")
    if level == 'CRITICAL':
        message = f"\n{'*' * (len(message) + 4)}\n* {message.upper()} *\n{'*' * (len(message) + 4)}\n\n"

    logger.log(getattr(logging, level, HINT_LEVEL if level == 'HINT' else None), colored_string(message, color))


def colored_string(string: any, color: str) -> Optional[str]:
    """
    :Description:
    Converts any input to a string and applies the specified color using ANSI escape codes.

    :param string: The input to be converted to a string and colored.
    :param color: The color to apply, specified as a string (e.g., 'RED', 'BRIGHT_GREEN').
    :return: The colored string, or None if an invalid color is provided.
    :rtype: Optional[str]
    """
    colors = {
        "RESET": "30",
        "RED": "31",
        "GREEN": "32",
        "YELLOW": "33",
        "BLUE": "34",
        "MAGENTA": "35",
        "CYAN": "36",
        "BLACK": "37",
        "BRIGHT_BLACK": "1;90",
        "BRIGHT_RED": "1;91",
        "BRIGHT_GREEN": "1;92",
        "BRIGHT_YELLOW": "1;93",
        "BRIGHT_BLUE": "1;94",
        "BRIGHT_MAGENTA": "1;95",
        "BRIGHT_CYAN": "1;96",
        "BRIGHT_WHITE": "1;97"
    }
    string = str(string)

    cc = colors.get(color.upper())
    if cc:
        begin = f"\033[{cc}m"
        end = "\033[m"

        return begin + string + end
    else:
        colors_to_select_str = ""
        for col in colors:
            colors_to_select_str += colored_string(col, col) + ' '

        show_error(f"Unproper color: '{colored_string(color, "BRIGHT_WHITE")}' in string: '{string}'\n"
                   f"You can select from colors:\n"
                   f"{colored_string('* ', "BRIGHT_WHITE") * (int((visible_length(colors_to_select_str) + 4) / 2))}\n"
                   f"{colored_string("* ", "BRIGHT_WHITE")}{colors_to_select_str}{colored_string("*", "BRIGHT_WHITE")}\n"
                   f"{colored_string('* ', "BRIGHT_WHITE") * (int((visible_length(colors_to_select_str) + 4) / 2))}")


def visible_length(s: str) -> int:
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return len(ansi_escape.sub('', s))


def show_error(error_description: str) -> None:
    """
    :Description:
    Shows error in logger (terminal)

    :param error_description: Description of an error
    :rtype: None
    """
    colored_log("ERROR", colored_string(error_description, 'BRIGHT_RED'))
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    error_messages = []

    while caller_frame:
        filename = caller_frame.f_code.co_filename
        lineno = caller_frame.f_lineno
        funcname = caller_frame.f_code.co_name
        code_line = linecache.getline(filename, lineno).strip()
        error_message = (f"\nIn file {filename}:{lineno}, function '{funcname}'\n"
                         f"-----> \t {code_line}")
        error_messages.append(error_message)
        caller_frame = caller_frame.f_back

    for message in reversed(error_messages):
        colored_log("ERROR", message)

    sys.exit(1)


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
