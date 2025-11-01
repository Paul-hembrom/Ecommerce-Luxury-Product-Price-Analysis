"""
Comprehensive logging utility for the Farfetch Pricing Analytics Engine.
Provides consistent logging configuration across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os
from datetime import datetime


def setup_logger(
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
        propagate: bool = False
) -> logging.Logger:
    """
    Set up and configure a logger with consistent settings.

    Args:
        name (str): Logger name, typically __name__ of the calling module
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file (str, optional): Path to log file. If None, logs to console only
        format_string (str, optional): Custom format string for log messages
        propagate (bool): Whether to propagate messages to parent loggers

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Application started")
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Default format string
    if format_string is None:
        format_string = (
            '%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s'
        )

    formatter = logging.Formatter(format_string)

    # Console handler (always added)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_project_logger(
        module_name: str,
        log_level: str = "INFO",
        log_to_file: bool = True
) -> logging.Logger:
    """
    Get a pre-configured logger for project modules with standardized settings.

    Args:
        module_name (str): Name of the module requesting the logger
        log_level (str): Log level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_to_file (bool): Whether to write logs to file

    Returns:
        logging.Logger: Configured logger instance
    """
    # Convert string level to logging constant
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    level = level_mapping.get(log_level.upper(), logging.INFO)

    # Determine log file path
    log_file = None
    if log_to_file:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = logs_dir / f"pricing_analytics_{timestamp}.log"

    # Create module-specific format
    format_string = (
        '%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s'
    )

    return setup_logger(
        name=module_name,
        level=level,
        log_file=str(log_file) if log_file else None,
        format_string=format_string,
        propagate=False
    )


class PerformanceLogger:
    """
    Context manager for logging performance of code blocks.

    Example:
        >>> with PerformanceLogger("data_processing"):
        ...     # Your code here
        ...     process_data()
    """

    def __init__(self, operation_name: str, logger: logging.Logger):
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(
                f"Completed operation: {self.operation_name} "
                f"(duration: {duration:.2f}s)"
            )
        else:
            self.logger.error(
                f"Operation failed: {self.operation_name} "
                f"(duration: {duration:.2f}s, error: {exc_val})"
            )


def log_execution_time(func):
    """
    Decorator to log function execution time.

    Example:
        >>> @log_execution_time
        ... def train_model(data):
        ...     # Model training code
        ...     pass
    """

    def wrapper(*args, **kwargs):
        logger = setup_logger(f"{func.__module__}.{func.__name__}")
        start_time = datetime.now()

        logger.debug(f"Function {func.__name__} started")

        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.debug(
                f"Function {func.__name__} completed "
                f"(duration: {duration:.2f}s)"
            )
            return result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.error(
                f"Function {func.__name__} failed "
                f"(duration: {duration:.2f}s, error: {str(e)})"
            )
            raise

    return wrapper


class DataQualityLogger:
    """
    Specialized logger for data quality checks and validations.
    """

    def __init__(self, module_name: str):
        self.logger = get_project_logger(f"data_quality.{module_name}")

    def log_missing_values(self, column: str, missing_count: int, total_count: int):
        """Log missing value statistics."""
        missing_pct = (missing_count / total_count) * 100
        if missing_pct > 10:
            self.logger.warning(
                f"High missing values in {column}: "
                f"{missing_count}/{total_count} ({missing_pct:.1f}%)"
            )
        else:
            self.logger.info(
                f"Missing values in {column}: "
                f"{missing_count}/{total_count} ({missing_pct:.1f}%)"
            )

    def log_data_validation(self, check_name: str, passed: bool, details: str = ""):
        """Log data validation results."""
        if passed:
            self.logger.info(f"Validation passed: {check_name} - {details}")
        else:
            self.logger.error(f"Validation failed: {check_name} - {details}")

    def log_data_summary(self, df_summary: dict):
        """Log data summary statistics."""
        self.logger.info("Data Summary:")
        for key, value in df_summary.items():
            self.logger.info(f"  {key}: {value}")


def configure_root_logger(level: int = logging.INFO, log_file: str = None):
    """
    Configure the root logger for the entire application.

    Args:
        level (int): Logging level
        log_file (str, optional): Path to log file
    """
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Basic configuration
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        handlers=[]
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    logging.root.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        logging.root.addHandler(file_handler)


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    test_logger = setup_logger("logger_test")
    test_logger.info("Logger test started")

    # Test different levels
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")

    # Test performance logging
    with PerformanceLogger("test_operation", test_logger):
        import time

        time.sleep(0.1)  # Simulate work

    # Test data quality logger
    dq_logger = DataQualityLogger("test_module")
    dq_logger.log_missing_values("price", 5, 100)
    dq_logger.log_data_validation("price_range", True, "All prices positive")

    test_logger.info("Logger test completed successfully")