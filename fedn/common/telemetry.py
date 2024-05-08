import os
import platform

import psutil
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
    OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from fedn.common.log_config import logger


class NoopTracer:
    def start_as_current_span(self, name, context=None, kind=None):
        return NoopSpan()


class NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_attribute(self, key, value):
        pass

    def add_event(self, name, attributes=None):
        pass

    def end(self, end_time=None):
        pass

    def record_exception(self, exception, attributes=None):
        pass

    def is_recording(self):
        return False

    def __call__(self, func):
        # Makes NoopSpan callable, and it returns the function itself, hence preserving its original functionality
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper


def initialize_tracer():
    if os.getenv("FEDN_TELEMETRY", 'true').lower() in ('false', '0'):
        return NoopTracer()
    else:
        logger.info("Telemetry enabled. Disable by setting FEDN_TELEMETRY=false")

    telemetry_server = os.getenv("FEDN_TELEMETRY_SERVER", 'https://telemetry.fedn.scaleoutsystems.com')
    telemetry_port = os.getenv("FEDN_TELEMETRY_PORT", 443)
    telemetry_service_name = os.getenv("FEDN_TELEMETRY_SERVICE_NAME", "FEDn")

    # Configure the tracer to report data to Jaeger
    trace.set_tracer_provider(
        TracerProvider(
            resource=Resource.create({SERVICE_NAME: telemetry_service_name})
        )
    )

    otlp_exporter = OTLPSpanExporter(
        endpoint=f"{telemetry_server}:{telemetry_port}",  # Default OTLP port for Jaeger, adjust as needed
    )

    # Attach the exporter to the tracer provider
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )

    return trace.get_tracer(__name__)


def get_context():
    context = {
                "fedn": {
                    "version": "0.9.1",
                },
                "hw": {
                    "cpu_count": os.cpu_count(),
                    "total_memory": psutil.virtual_memory().total,
                    "available_memory": psutil.virtual_memory().available,
                    "total_disk_space": psutil.disk_usage('/').total,
                    "available_disk_space": psutil.disk_usage('/').free,
                },
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "platform": platform.platform(),
                    "python_implementation": platform.python_implementation(),
                    "python_version": platform.python_version(),
                    "machine": platform.machine(),
                    "architecture": platform.architecture(),
                    "version": platform.uname().version,
                },
            }
    return context


# Initialize tracer
tracer = initialize_tracer()
try:
    with tracer.start_as_current_span("initialize_tracer") as span:
        context = get_context()
        span.set_attribute("context", str(context))
except Exception as e:
    logger.error("Failed to initialize tracer: {}".format(e))


def trace_all_methods(cls):
    def traced_method(method):
        """Wrap the method so that it executes within a traced span."""
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(method.__name__) as span:
                # Set the class name attribute on the span
                span.set_attribute("class_name", cls.__name__)
                return method(*args, **kwargs)
        return wrapper

    # Apply the traced_method decorator to each callable method of the class
    for key, method in cls.__dict__.items():
        if callable(method) and not key.startswith('__'):
            # Set the decorated method back on the class
            setattr(cls, key, traced_method(method))
    return cls
