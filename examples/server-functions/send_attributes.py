import argparse
import json
import random
import time

from scaleout import APIClient


def parse_args():
    parser = argparse.ArgumentParser(description="Send a random 'charging' attribute to the controller for a client.")
    parser.add_argument("--api-url", required=True, help="Base URL of the API server (the same api-url that is used to connect clients)")
    parser.add_argument("--admin-token", required=True, help="Authentication token for the API (admin token which can be generated from the studio UI)")
    parser.add_argument("--client-id", required=True, help="client ID for the current client (the same api-url that is used to connect the client)")
    parser.add_argument("--delay", required=False, default=30, help="Delay between sending attributes.")
    return parser.parse_args()


def main():
    # --------- NOTE ------------
    # run this script with --token <admin token> (fetch admin token from studio)
    # if this script is not running the server functions example will default to picking clients.
    args = parse_args()
    client = APIClient(host=args.api_url, token=args.admin_token, secure=True, verify=True)
    while True:
        # Prepare a random charging status
        attribute_payload = {"key": "charging", "value": random.choice(["True", "False"]), "sender": {"name": "", "role": "", "client_id": args.client_id}}

        # Send to server
        try:
            result = client.add_attributes(attribute_payload)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Failed to send attributes: {e}")
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
