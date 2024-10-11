Creating Your Own Client
========================

This guide will help you create your own client for the FEDn network.

Step-by-Step Instructions
-------------------------

1. **Create a virutal environment**: Start by creating a virtual environment and activating it.

    ```bash
    python3 -m venv fedn-env
    source fedn-env/bin/activate
    ```

2. **Install FEDn**: Install the FEDn package by running the following command:

    ```bash
    pip install fedn
    ```

3. **Create your client**: Copy and paste the code Below into a new file called `client_example.py`.

    ```python
    import argparse
    import json
    import threading
    import uuid

    import fedn.network.grpc.fedn_pb2 as fedn
    from fedn.network.clients.client_api import ClientAPI, ConnectToApiResult

    client_api = ClientAPI()


    def main(api_url: str, api_port: int, token: str = None, name: str = None):
        print(f"API URL: {api_url}")
        print(f"API Token: {token or "-"}")
        print(f"API Port: {api_port or "-"}")

        if name is None:
            name = input("Enter Client Name: ")

        url = f"{api_url}:{api_port}" if api_port else api_url

        if not url.endswith("/"):
            url += "/"

        print(f"Client Name: {name}")

        client_id = str(uuid.uuid4())

        print("Connecting to API...")

        client_options = {
            "name": "client_example",
            "client_id": client_id,
            "package": "local",
            "preferred_combiner": "",
        }

        result, combiner_config = client_api.connect_to_api(url, token, client_options)

        if result != ConnectToApiResult.Assigned:
            print("Failed to connect to API, exiting.")
            return

        print("Connected to API")

        result: bool = client_api.init_grpchandler(config=combiner_config, client_name=client_id, token=token)

        if not result:
            return

        threading.Thread(target=client_api.send_heartbeats, kwargs={"client_name": name, "client_id": client_id}, daemon=True).start()

        def on_train(request):
            print("Received train request")
            model_id: str = request.model_id

            model = client_api.get_model_from_combiner(id=str(model_id), client_name=name)

            # Do your training here
            out_model = model
            updated_model_id = uuid.uuid4()
            client_api.send_model_to_combiner(out_model, str(updated_model_id))

            #         val metadataJson = buildJsonObject {
            #     put("num_examples", 1)
            #     put("batch_size", 1)
            #     put("epochs", 1)
            #     put("lr", 1)
            # }
            # val configJson = buildJsonObject {
            #     put("round_id", 1)
            # }

            # val json = buildJsonObject {
            #     put("training_metadata", metadataJson)
            #     put("config", configJson.toString())
            # }

            training_metadata = {
                "num_examples": 1,
                "batch_size": 1,
                "epochs": 1,
                "lr": 1,
            }

            config = {
                "round_id": 1,
            }

            client_api.send_model_update(
                sender_name=name,
                sender_role=fedn.WORKER,
                client_id=client_id,
                model_id=model_id,
                model_update_id=str(updated_model_id),
                receiver_name=request.sender.name,
                receiver_role=request.sender.role,
                meta={
                    "training_metadata": training_metadata,
                    "config": json.dumps(config),
                },
            )

        client_api.subscribe("train", on_train)

        threading.Thread(target=client_api.listen_to_task_stream, kwargs={"client_name": name, "client_id": client_id}, daemon=True).start()

        stop_event = threading.Event()
        try:
            stop_event.wait()
        except KeyboardInterrupt:
            print("Client stopped by user.")


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Client Example")
        parser.add_argument("--api-url", type=str, required=True, help="The API URL")
        parser.add_argument("--api-port", type=int, required=False, help="The API Port")
        parser.add_argument("--token", type=str, required=False, help="The API Token")
        parser.add_argument("--name", type=str, required=False, help="The Client Name")

        args = parser.parse_args()
        main(args.api_url, args.api_port)

    ```
4. **Run the client**: Run the client by executing the following command:

    ```bash
    python client_example.py --api-url http(s)://<api-url> --token <api-token>
    ```
    Replace `<api-url>` and `<api-token>` with the URL and token of the FEDn API.

5. **Start training**: Create a session and start training by using either the FEDn CLI or the FEDn UI.