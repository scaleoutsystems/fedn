import json
from pathlib import Path

# Environment configurations
LOCAL_CONFIG = {
    "DISCOVER_HOST": "127.0.0.1",
    "DISCOVER_PORT": 8092,
    "IS_LOCAL": True,
    "IS_REFERENCE": False,
    "SECURE": False,
    "VERIFY": False,
    "CLIENT_TOKEN": None,
    "ADMIN_TOKEN": None,
    "COMBINER_PREFFERRED": "",
}

REMOTE_CONFIG = {
    "DISCOVER_HOST": "https://<your-discover-host>",
    "NodeIP": "<your-node-ip>",
    "DISCOVER_PORT": "<your-discover-port>",
    "IS_LOCAL": False,
    "IS_REFERENCE": False,
    "SECURE": True,
    "VERIFY": True,
    "COMBINER_PREFFERRED": "",
    "CLIENT_TOKEN":  "<your-client-token>",
    "ADMIN_TOKEN": "<your-admin-token>",
}

REFERENCE_CONFIG = {
    "DISCOVER_HOST": "https://<your-discover-host>",
    "NodeIP": "<your-node-ip>",
    "DISCOVER_PORT": "<your-discover-port>", # NodePort 
    "IS_LOCAL": False,
    "IS_REFERENCE": True,
    "SECURE": True,
    "VERIFY": True,
    "CLIENT_TOKEN": "<your-client-token>",
    "ADMIN_TOKEN": "<your-admin-token>",
    "COMBINER_PREFFERRED": "", # When running with reference setup, you can specify a preferred combiner here for the separate VMs to distribute the load
}

# Common settings that don't change between environments
COMMON_SETTINGS = {
    "N_CLIENTS": 29,
    "N_EPOCHS": 10,
    "N_ROUNDS": 100,
    "N_SESSIONS": 6,
    "N_CYCLES": 30,
    "CLIENTS_MAX_DELAY": 30,
    "CLIENTS_ONLINE_FOR_SECONDS": 120,
    "ROUND_TIMEOUT": 90,
}

# Choose which environment to use
USE_LOCAL = False  # Set to False to use remote environment
USE_REFERENCE = False  # Set to True to use reference configuration

if USE_REFERENCE:
    settings = {**COMMON_SETTINGS, **REFERENCE_CONFIG}
else:
    settings = {**COMMON_SETTINGS, **(LOCAL_CONFIG if USE_LOCAL else REMOTE_CONFIG)}

# Combine the selected environment config with common settings


