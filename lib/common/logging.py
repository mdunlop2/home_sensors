import logging


def get_logger(name: str):
    """
    Common logger configuration
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(name)
