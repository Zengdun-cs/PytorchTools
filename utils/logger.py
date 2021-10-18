import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.getLogger().setLevel(logging.INFO)


class Logger(object):
    """record cmd info to file and print it to cmd at the same time
    
    Args:
        log_name (str): log name for output.
        log_file (str): a file path of log file.
    """
    def __init__(self, log_name=None, log_file=None):
        if log_name is not None:
            self.logger = logging.getLogger(log_name)
            self.name = log_name
        else:
            logging.getLogger().setLevel(logging.INFO)
            self.logger = logging
            self.name = "root"

        if log_file is not None:
            handler = logging.FileHandler(log_file, mode='w')
            handler.setLevel(level=logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, log_str):
        """Print information to logger"""
        self.logger.info(log_str)

    def warning(self, warning_str):
        """Print warning to logger"""
        self.logger.warning(warning_str)
