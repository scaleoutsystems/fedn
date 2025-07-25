import json
from pathlib import Path

# Environment configurations
LOCAL_CONFIG = {
    "DISCOVER_HOST": "127.0.0.1",
    "DISCOVER_PORT": 8092,
    "IS_LOCAL": True,
    "SECURE": False,
    "VERIFY": False,
    "CLIENT_TOKEN": None,
    "ADMIN_TOKEN": None,
}

REMOTE_CONFIG = {
    "DISCOVER_HOST": "https://<your-discover-host>",
    "NodeIP": "<your-node-ip>",
    "DISCOVER_PORT": "<your-discover-port>",
    "IS_LOCAL": False,
    "SECURE": True,
    "VERIFY": True,
    "CLIENT_TOKEN": "<your-client-token>",
    "ADMIN_TOKEN": "<your-admin-token>",
}

# Common settings that don't change between environments
COMMON_SETTINGS = {
    "N_CLIENTS": 500,
    "N_EPOCHS": 10,
    "N_ROUNDS": 100,
    "N_SESSIONS": 6,
    "N_CYCLES": 30,
    "CLIENTS_MAX_DELAY": 30,
    "CLIENTS_ONLINE_FOR_SECONDS": 120,
}

# Choose which environment to use
USE_LOCAL = False  # Set to False to use remote environment

# Combine the selected environment config with common settings
settings = {**COMMON_SETTINGS, **(LOCAL_CONFIG if USE_LOCAL else REMOTE_CONFIG)}

