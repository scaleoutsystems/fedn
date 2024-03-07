import json
import logging
import logging.config
import os
import threading
from functools import wraps

import psutil
import requests
import urllib3

try:
    
    import platform
    import socket

    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes

    telemetry_enabled = True
except ImportError:
    telemetry_enabled = False

# def get_system_info():
#     system_info = [
#         ["os.name", os.name],
#         ["platform.system", platform.system()],
#         ["platform.release", platform.release()],
#         ["hostname", socket.gethostname()],
#         ["ip_address", socket.gethostbyname(socket.gethostname())],
#         ["cpu_count", psutil.cpu_count(logical=True)],
#         ["total_memory", psutil.virtual_memory().total],
#         ["total_disk", psutil.disk_usage('/').total],
#     ]
#     return system_info


# # Configure the tracer to export traces to Jaeger
# resource = Resource.create({ResourceAttributes.SERVICE_NAME: "FEDn Client"})
# tracer_provider = TracerProvider(resource=resource)
# trace.set_tracer_provider(tracer_provider)

# # Create a JaegerExporter
# jaeger_exporter = JaegerExporter(
#     agent_host_name='localhost',
#     agent_port=6831,
# )

# # Add the Jaeger exporter to the tracer provider
# tracer_provider.add_span_processor(
#     BatchSpanProcessor(jaeger_exporter)
# )

# tracer = None



urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)

handler = logging.StreamHandler()
logger = logging.getLogger("fedn")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

# def add_trace(name=""):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             self = args[0]
#             name = func.__name__
#             if tracer:
                
#                 with tracer.start_as_current_span(name) as span:
#                     # print("name={}....{}".format(name, attributes))
#                     if self.trace_attribs:
#                         for attrib in self.trace_attribs:
#                             span.set_attribute(attrib[0], attrib[1])
#                         # system_attribs = get_system_info()
#                         # print(system_attribs)
#                         # for attrib in system_attribs:
#                         #     span.set_attribute(attrib[0], attrib[1])
#                     return func(*args, **kwargs)
#             else:
#                 return func(*args, **kwargs)
#         return wrapper
#     return decorator


def get_tracer():
    global tracer
    return tracer


def enable_tracing():
    global tracer
    tracer = trace.get_tracer(__name__)

def get_disk_usage(path='/'):
    disk_usage = psutil.disk_usage(path)
    
    total_gb = disk_usage.total / (1024**3)
    used_gb = disk_usage.used / (1024**3)
    free_gb = disk_usage.free / (1024**3)
    percent_used = disk_usage.percent
    
    return total_gb, used_gb, free_gb, percent_used


def get_network_io_in_gb():
    net_io = psutil.net_io_counters()
    bytes_sent = net_io.bytes_sent / (1024**3)  # Convert from bytes to GB
    bytes_recv = net_io.bytes_recv / (1024**3)  # Convert from bytes to GB

    return bytes_sent, bytes_recv


def periodic_function():
    print("This function is called every 5 seconds.")
    # Get the current process ID
    pid = os.getpid()

    # Get the process instance for the current process
    p = psutil.Process(pid)

    # CPU usage of current process
    cpu_usage = p.cpu_percent(interval=0.01)

    # Memory usage of current process
    memory_usage = p.memory_info().rss / (1024 * 1024 * 1024)  # Convert bytes to GB

    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage} GB")
    total_gb, used_gb, free_gb, percent_used = get_disk_usage('/')

    print(f"Total: {total_gb:.2f} GB")
    print(f"Used: {used_gb:.2f} GB")
    print(f"Free: {free_gb:.2f} GB")
    print(f"Percent Used: {percent_used}%")

    bytes_sent, bytes_recv = get_network_io_in_gb()

    print(f"GB Sent: {bytes_sent:.2f} GB")
    print(f"GB Received: {bytes_recv:.2f} GB")
    # Schedule this function to be called again after 5 seconds
    threading.Timer(5, periodic_function).start()


class CustomHTTPHandler(logging.handlers.HTTPHandler):
    def __init__(self, host, url, method='POST', credentials=None, projectname='', apptype=''):
        super().__init__(host, url, method)
        self.credentials = credentials  # Basic auth (username, password)
        self.projectname = projectname
        self.apptype = apptype

    def emit(self, record):
        # Customize the log record, for example, by adding metadata
        # record.projectname = self.projectname
        # record.apptype = self.apptype

        # Convert log record to json format
        
        log_entry = self.mapLogRecord(record)

        log_entry = {
            "msg": log_entry['msg'],
            "levelname": log_entry['levelname'],
            "project": os.environ.get("PROJECT_ID"),
            "appinstance": os.environ.get("APP_ID")

        }
        # Setup headers
        headers = {
            'Content-type': 'application/json',
        }
        if self.credentials:
            import base64
            auth = base64.b64encode(f"{self.credentials[0]}:{self.credentials[1]}".encode('utf-8')).decode('utf-8')
            headers['Authorization'] = f'Basic {auth}'

        # Use http.client or requests to send the log data
        if self.method.lower() == 'post':
            requests.post(self.host+self.url, json=log_entry, headers=headers)
        else:
            # Implement other methods if needed, e.g., GET
            pass


def log_remote(server='http://studio-studio:8080', path='/api/applog/'):
    # http_handler = logging.handlers.HTTPHandler(server, '/log', method='POST')
    http_handler = CustomHTTPHandler(
        host=server,
        url=path,
        method='POST',
        credentials=None,  # Basic Auth
        projectname='test-project',
        apptype='client'
    )
    http_handler.setLevel(logging.DEBUG)
    logger.addHandler(http_handler)

log_remote()

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
