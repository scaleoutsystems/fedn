import argparse
from fedn import APIClient

from data import get_dataloaders
from model import compile_model, model_hyperparams, save_parameters
from settings import DATASET_PATH, KEYWORDS, BATCHSIZE_VALID


HOST = ""   ## INSERT HOST
TOKEN = ""  ## INSERT TOKEN

def init_seedmodel(api_client: APIClient):

    # Use all data to compute seed model normalizers for now
    dataloader_train, _, _ = get_dataloaders(DATASET_PATH, KEYWORDS, 0,
                                            1, BATCHSIZE_VALID, BATCHSIZE_VALID)
    sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
    seed_path = "seed.npz"
    save_parameters(sc_model, seed_path)

    response = api_client.set_active_model(seed_path)

    return response


def main():
    parser = argparse.ArgumentParser(description="API example")
    parser.add_argument("--init-seed", action="store_true", required=False, help="Use this flag to send a seed model to the server")
    parser.add_argument("--start-session", action="store_true", required=False, help="Use this flag to start session")
    args = parser.parse_args()

    if HOST == "" and TOKEN == "":
        print("Please insert HOST and TOKEN in fedn_api.py")
        return

    api_client = APIClient(host=HOST, secure=True, verify=True, token=TOKEN)

    if args.init_seed:
        response = init_seedmodel(api_client)
        print(response)
    elif args.start_session:
        #Depending on the computer hosting the clients this round_timeout might need to increase
        response = api_client.start_session(round_timeout=600)
        print(response)
    else:
        print("No flag passed")



if __name__ == "__main__":
    main()
