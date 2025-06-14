import logging
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger configured to log to both console and a file in `.log/` directory.
    """

    # Ensure the log directory exists
    log_dir = ".log"
    os.makedirs(log_dir, exist_ok=True)

    # Log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if logger already exists
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    # Example usage
    get_logger( "message_logger")