import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = f"{ROOT}/logs/app.log"



def get_logger(name: str = "Skyguard") -> logging.Logger:
    logger = logging.getLogger(name=name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='[%(asctime)s]:[%(lineno)d]:[%(filename)s]:[%(module)s]:message: %(message)s',
        datefmt= "%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = get_logger()
