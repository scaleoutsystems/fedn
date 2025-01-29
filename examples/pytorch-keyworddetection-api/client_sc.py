from torch import nn
from torch.optim import Adam

import argparse
import io
from pathlib import Path

from torcheval.metrics import MulticlassAccuracy

from fedn.network.clients.fedn_client import FednClient, ConnectToApiResult

from data import get_dataloaders
from model import model_hyperparams, compile_model, load_parameters, save_parameters
from settings import KEYWORDS, BATCHSIZE_VALID, DATASET_PATH, DATASET_TOTAL_SPLITS
from util import construct_api_url, read_settings



def main():
    parser = argparse.ArgumentParser(description="Client SC Example")
    parser.add_argument("--client-yaml", type=str, required=False, help="Settings specfic for the client (default: client.yaml)")
    parser.add_argument("--dataset-split-idx", type=int, required=True, help="Setting for which dataset split this client uses")

    parser.set_defaults(client_yaml="client.yaml")
    args = parser.parse_args()

    start_client(args.client_yaml, args.dataset_split_idx)



def start_client(client_yaml, dataset_split_idx):
    #Constant in this context
    DATASET_SPLIT_IDX = dataset_split_idx

    if Path(client_yaml).exists():
            cfg = read_settings(client_yaml)
    else:
        raise Exception(f"Client yaml file not found: {client_yaml}")

    if "discover_host" in cfg:
        cfg["api_url"] = cfg["discover_host"]

    url = construct_api_url(cfg["api_url"], cfg.get("api_port", None))

    def on_train(model_params, settings):
        training_metadata = {"batchsize_train":64, "lr": 1e-3, "n_epochs":1}
        print(settings)


        dataloader_train, _, _ = get_dataloaders(DATASET_PATH, KEYWORDS, DATASET_SPLIT_IDX,
                                                DATASET_TOTAL_SPLITS, training_metadata["batchsize_train"],
                                                BATCHSIZE_VALID)

        sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
        load_parameters(sc_model, model_params)
        optimizer = Adam(sc_model.parameters(), lr=training_metadata["lr"])
        loss_fn = nn.CrossEntropyLoss()
        n_epochs = training_metadata["n_epochs"]

        for epoch in range(n_epochs):
            sc_model.train()
            for idx, (y_labels, x_spectrograms) in enumerate(dataloader_train):
                optimizer.zero_grad()
                _, logits =  sc_model(x_spectrograms)

                loss = loss_fn(logits, y_labels)
                loss.backward()

                optimizer.step()

                if idx%100 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs} | Batch: {idx+1}/{len(dataloader_train)} | Loss: {loss.item()}")

        out_model = save_parameters(sc_model, io.BytesIO())

        metadata = { "training_metadata": {"num_examples": len(dataloader_train.dataset)} }

        return out_model, metadata

    def on_validate(model_params):
        dataloader_train, dataloader_valid, dataloader_test = get_dataloaders(DATASET_PATH, KEYWORDS, DATASET_SPLIT_IDX,
                                                                                DATASET_TOTAL_SPLITS, BATCHSIZE_VALID, BATCHSIZE_VALID)

        n_labels = dataloader_train.dataset.n_labels

        sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
        load_parameters(sc_model, model_params)

        def evaluate(dataloader):
            sc_model.eval()
            metric = MulticlassAccuracy(num_classes=n_labels)
            for (y_labels, x_spectrograms) in dataloader:

                probs, _ =  sc_model(x_spectrograms)

                y_pred = probs.argmax(-1)
                metric.update(y_pred, y_labels)
            return metric.compute().item()

        report = {"training_acc": evaluate(dataloader_train),
                  "validation_acc": evaluate(dataloader_valid)}

        return report

    def on_predict(model_params):
        dataloader_train, _, _ = get_dataloaders(DATASET_PATH, KEYWORDS, DATASET_SPLIT_IDX,
                                                  DATASET_TOTAL_SPLITS, BATCHSIZE_VALID, BATCHSIZE_VALID)
        sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
        load_parameters(sc_model, model_params)

        return {}

    fedn_client = FednClient(train_callback=on_train, validate_callback=on_validate, predict_callback=on_predict)


    fedn_client.set_name(cfg["name"])
    fedn_client.set_client_id(cfg["client_id"])

    controller_config = {
        "name": fedn_client.name,
        "client_id": fedn_client.client_id,
        "package": "local",
        "preferred_combiner": "",
    }

    result, combiner_config = fedn_client.connect_to_api(url, cfg["token"], controller_config)

    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to API, exiting.")
        return

    result: bool = fedn_client.init_grpchandler(config=combiner_config, client_name=fedn_client.client_id, token=cfg["token"])

    if not result:
        return

    fedn_client.run()



if __name__ == "__main__":
    main()
