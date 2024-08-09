from abc import ABC


class Config(ABC):
    pass


class ReducerConfig(Config):
    """Configuration for the Reducer component."""

    compute_bundle_dir = None
    models_dir = None

    initial_model = None

    storage_backend = {"type": "s3", "settings": {"bucket": "models"}}

    def __init__(self):
        pass
