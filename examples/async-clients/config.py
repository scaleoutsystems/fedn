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
    "DISCOVER_HOST": "api.studio.scaleoutplatform.com/asyncclitest-zmh-fedn-reducer",
    "DISCOVER_PORT": None,
    "IS_LOCAL": False,
    "SECURE": True,
    "VERIFY": True,
    "CLIENT_TOKEN": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MDkyODgyLCJpYXQiOjE3NDU1MDA4ODIsImp0aSI6IjUxZGUxNzhiN2Y4OTQ3ZWJiYjNkNTg0ODYyNzBmYTFmIiwidXNlcl9pZCI6NTgsImNyZWF0b3IiOiJzaWd2YXJkQHNjYWxlb3V0c3lzdGVtcy5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXN5bmNjbGl0ZXN0LXptaCJ9.lhnb-7n80fqsprKuF5M4qOdVAlJlsaXEgXG_yAY0n10",
    "ADMIN_TOKEN": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MDkyODY3LCJpYXQiOjE3NDU1MDA4NjcsImp0aSI6ImU4NDFjMjVlZDM2NTRlNDc4NmIxN2E5Yjg0MzU0NjM5IiwidXNlcl9pZCI6NTgsImNyZWF0b3IiOiJzaWd2YXJkQHNjYWxlb3V0c3lzdGVtcy5jb20iLCJyb2xlIjoiYWRtaW4iLCJwcm9qZWN0X3NsdWciOiJhc3luY2NsaXRlc3Qtem1oIn0.xa9r413N_FyGxo7kvG_8iGlSf1z-LnJucoF41aRXris",
}

# Common settings that don't change between environments
COMMON_SETTINGS = {
    "N_CLIENTS": 200,
    "N_EPOCHS": 10,
    "N_ROUNDS": 100,
    "N_SESSIONS": 6,
    "N_CYCLES": 30,
    "CLIENTS_MAX_DELAY": 10,
    "CLIENTS_ONLINE_FOR_SECONDS": 120,
}

# Choose which environment to use
USE_LOCAL = True  # Set to False to use remote environment

# Combine the selected environment config with common settings
settings = {**COMMON_SETTINGS, **(LOCAL_CONFIG if USE_LOCAL else REMOTE_CONFIG)}

# Only try to load tokens for remote configuration
if not USE_LOCAL:
    tokens_file = Path(__file__).parent / "tokens.json"
    if tokens_file.exists():
        try:
            with open(tokens_file, "r") as f:
                tokens = json.load(f)

            # Use the discover host as the key to find the right tokens
            discover_host = settings["DISCOVER_HOST"]
            if discover_host in tokens:
                settings.update({k: v for k, v in tokens[discover_host].items() if k in settings})
            else:
                print(f"Warning: No tokens found for host '{discover_host}' in tokens.json")
        except Exception as e:
            print(f"Warning: Could not load tokens from {tokens_file}: {e}")
    else:
        print(f"Warning: No tokens file found at {tokens_file}. Required for remote configuration.")
