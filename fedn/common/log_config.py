import logging
import logging.config
import os

import requests
import urllib3
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import \
    OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)

handler = logging.StreamHandler()
logger = logging.getLogger("fedn")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)


class StudioHTTPHandler(logging.handlers.HTTPHandler):
    def __init__(self, host, url, method="POST", token=None):
        super().__init__(host, url, method)
        self.token = token

    def emit(self, record):
        log_entry = self.mapLogRecord(record)

        log_entry = {
            "msg": log_entry["msg"],
            "levelname": log_entry["levelname"],
            "project": os.environ.get("PROJECT_ID"),
            "appinstance": os.environ.get("APP_ID"),
        }
        headers = {
            "Content-type": "application/json",
        }
        if self.token:
            remote_token_protocol = os.environ.get("FEDN_REMOTE_LOG_TOKEN_PROTOCOL", "Token")
            headers["Authorization"] = f"{remote_token_protocol} {self.token}"
        if self.method.lower() == "post":
            requests.post(self.host + self.url, json=log_entry, headers=headers)


# Remote logging configuration
REMOTE_LOG_SERVER = os.environ.get("FEDN_REMOTE_LOG_SERVER", False)
REMOTE_LOG_PATH = os.environ.get("FEDN_REMOTE_LOG_PATH", False)
REMOTE_LOG_LEVEL = os.environ.get("FEDN_REMOTE_LOG_LEVEL", "INFO")

if REMOTE_LOG_SERVER:
    rloglevel = log_levels.get(REMOTE_LOG_LEVEL, logging.INFO)
    remote_token = os.environ.get("FEDN_REMOTE_LOG_TOKEN", None)

    http_handler = StudioHTTPHandler(host=REMOTE_LOG_SERVER, url=REMOTE_LOG_PATH, method="POST", token=remote_token)
    http_handler.setLevel(rloglevel)
    logger.addHandler(http_handler)


# OpenTelemetry Logging Configuration
logger_provider = LoggerProvider()
set_logger_provider(logger_provider)
exporter = OTLPLogExporter()
logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

otel_handler = LoggingHandler(logger_provider=logger_provider)
logger.addHandler(otel_handler)


def set_log_level_from_string(level_str):
    """Set the log level based on a string input."""
    level_mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    level = level_mapping.get(level_str.upper())
    if not level:
        raise ValueError(f"Invalid log level: {level_str}")
    logger.setLevel(level)


def set_log_stream(log_file):
    """Redirect the log stream to a specified file, if log_file is set."""
    if not log_file:
        return
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def shutdown_logging():
    """Shutdown the logger provider to ensure logs are flushed before exit."""
    logger_provider.shutdown()
