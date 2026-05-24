import logging
import os
from datetime import datetime


def get_logger(name="CancerGrading", log_dir="outputs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(
        os.path.join(log_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
