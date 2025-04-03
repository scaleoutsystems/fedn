import argparse

from data import get_dataloaders
from model import compile_model, model_hyperparams, save_parameters
from settings import BATCHSIZE_VALID, DATASET_PATH, KEYWORDS

from fedn import APIClient

HOST = ""  ## INSERT HOST
TOKEN = ""  ## INSERT TOKEN


def init_seedmodel() -> dict:
    """Used to send a seed model to the server. The seed model is normalized with all training data."""
    dataloader_train, _, _ = get_dataloaders(DATASET_PATH, KEYWORDS, 0, 1, BATCHSIZE_VALID, BATCHSIZE_VALID)
    sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
    seed_path = "seed.npz"
    save_parameters(sc_model, seed_path)
    print("Seed saved to seed.npz")


def main():
    parser = argparse.ArgumentParser(description="API example")
    parser.add_argument("--init-seed", action="store_true", required=False, help="Use this flag to send a seed model to the server")
    parser.add_argument("--start-session", action="store_true", required=False, help="Use this flag to start session")
    args = parser.parse_args()

    if args.init_seed:
        init_seedmodel()
    elif args.start_session:
        if HOST == "" and TOKEN == "":
            print("Please insert HOST and TOKEN in fedn_api.py")
            return
        if TOKEN:
            api_client = APIClient(host=HOST, secure=True, verify=True, token=TOKEN)
        else:
            api_client = APIClient(host=HOST)
        # Depending on the computer hosting the clients this round_timeout might need to increase
        response = api_client.start_session(name="Training", round_timeout=1200)
        print(response)
    else:
        print("No flag passed")


if __name__ == "__main__":
    main()
