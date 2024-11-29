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
    "DISCOVER_HOST": "fedn.scaleoutsystems.com/<project-slug>",
    "DISCOVER_PORT": None,
    "IS_LOCAL": False,
    "SECURE": True,
    "VERIFY": True,
    "CLIENT_TOKEN": None,
    "ADMIN_TOKEN": None,
}

# Common settings that don't change between environments
COMMON_SETTINGS = {
    "N_CLIENTS": 10,
    "N_EPOCHS": 10,
    "N_ROUNDS": 50,
    "N_SESSIONS": 1,
    "N_CYCLES": 1,
    "CLIENTS_MAX_DELAY": 10,
    "CLIENTS_ONLINE_FOR_SECONDS": 120,
}

# Choose which environment to use
USE_LOCAL = True  # Set to False to use remote environment

# Combine the selected environment config with common settings
settings = {**COMMON_SETTINGS, **(LOCAL_CONFIG if USE_LOCAL else REMOTE_CONFIG)}
