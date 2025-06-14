# logger.py
import logging
import os
from datetime import datetime

def get_logger() -> logging.Logger:
    """
    Returns a logger configured to log to both console and a file in `.log/` directory.
    The logger is named after the calling module's __name__.
    """
    # Create logs directory if it doesn't exist
    log_dir = ".log"
    os.makedirs(log_dir, exist_ok=True)

    # Get the name of the current module (e.g., '__main__' or actual module name)
    logger_name = __name__

    # Generate timestamped log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{logger_name}_{timestamp}.log")

    # Get the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

get_logger()  # Initialize the logger when this module is imported