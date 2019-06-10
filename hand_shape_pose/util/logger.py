# Copyright (c) Liuhao Ge. All Rights Reserved.
import logging
import os
import sys
import time
import datetime

def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_logger_filename():
    ts = time.time()
    return 'log-' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S') + '.txt'
