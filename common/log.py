import logging


def init_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
