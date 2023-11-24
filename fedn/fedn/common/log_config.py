import logging
import logging.config
from functools import wraps

import urllib3

try:
    import os
    import platform
    import socket

    import psutil
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes

    telemetry_enabled = True
except ImportError:
    telemetry_enabled = False

def get_system_info():
    system_info = [
        ["os.name", os.name],
        ["platform.system", platform.system()],
        ["platform.release", platform.release()],
        ["hostname", socket.gethostname()],
        ["ip_address", socket.gethostbyname(socket.gethostname())],
        ["cpu_count", psutil.cpu_count(logical=True)],
        ["total_memory", psutil.virtual_memory().total],
        ["total_disk", psutil.disk_usage('/').total],
    ]
    return system_info

# Configure the tracer to export traces to Jaeger
resource = Resource.create({ResourceAttributes.SERVICE_NAME: "FEDn Client"})
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

# Create a JaegerExporter
jaeger_exporter = JaegerExporter(
    agent_host_name='localhost',
    agent_port=6831,
)

# Add the Jaeger exporter to the tracer provider
tracer_provider.add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = None



urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)

handler = logging.StreamHandler()
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

def add_trace(name=""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            name = func.__name__
            if tracer:
                
                with tracer.start_as_current_span(name) as span:
                    # print("name={}....{}".format(name, attributes))
                    if self.trace_attribs:
                        for attrib in self.trace_attribs:
                            span.set_attribute(attrib[0], attrib[1])
                        # system_attribs = get_system_info()
                        # print(system_attribs)
                        # for attrib in system_attribs:
                        #     span.set_attribute(attrib[0], attrib[1])
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def get_tracer():
    global tracer
    return tracer

def enable_tracing():
    global tracer
    tracer = trace.get_tracer(__name__)

def log_remote(server='localhost:8000', path='/log'):
    http_handler = logging.handlers.HTTPHandler(server, '/log', method='POST')
    http_handler.setLevel(logging.WARNING)
    logger.addHandler(http_handler)

def set_log_level_from_string(level_str):
    """
    Set the log level based on a string input.
    """
    # Mapping of string representation to logging constants
    level_mapping = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
    }

    # Get the logging level from the mapping
    level = level_mapping.get(level_str.upper())

    if not level:
        raise ValueError(f"Invalid log level: {level_str}")

    # Set the log level
    logger.setLevel(level)


def set_log_stream(log_file):
    """
    Redirect the log stream to a specified file, if log_file is set.
    """
    if not log_file:
        return

    # Remove existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Create a FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
