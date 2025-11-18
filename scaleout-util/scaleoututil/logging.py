import logging
import sys


class FednLogger:
    """A simple logger wrapper for FEDn."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FednLogger, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        self.logger = logging.getLogger("fedn")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.__initialized = True

    @property
    def _formatter(self):
        return logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        self.logger.critical(*args, **kwargs)

    def set_log_level_from_string(self, level_str):
        """Set the log level based on a string input."""
        level_str = level_str.upper()
        mapping = logging.getLevelNamesMapping()
        if level_str not in mapping:
            raise ValueError(f"Invalid log level: {level_str}")
        self.logger.setLevel(mapping[level_str])

    def set_log_stream(self, log_file=None):
        """Redirect logs to a file if log_file is provided, otherwise to stdout."""
        # Remove existing Stream and File handlers
        for handler in list(self.logger.handlers):
            if isinstance(handler, (logging.StreamHandler, logging.FileHandler)):
                self.logger.removeHandler(handler)
        if log_file:
            # Add a FileHandler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._formatter)
            self.logger.addHandler(file_handler)
        else:
            # Add a StreamHandler for stdout
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(self._formatter)
            self.logger.addHandler(stream_handler)

    def add_handler(self, handler):
        """Add a custom handler to the logger."""
        self.logger.addHandler(handler)

    def shutdown_logging(self):
        """Shutdown the logger to ensure logs are flushed before exit."""
        logging.shutdown()
