import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    # Create utilities/logs directory if it does not exist
    log_directory = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Path for the log file
    log_file_path = os.path.join(log_directory, 'app.log')

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the minimum level of log messages

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = RotatingFileHandler(log_file_path, maxBytes=1024*1024*5, backupCount=5)  # File handler

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Set level for handlers
    c_handler.setLevel(logging.DEBUG)  # Console only shows warning and above by default
    f_handler.setLevel(logging.DEBUG)  # File logs everything

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = setup_logging()
