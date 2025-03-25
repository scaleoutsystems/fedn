import click
from config import settings
from fedn import APIClient


def init_fedn(seed_path):
    client = APIClient(
        host=settings["DISCOVER_HOST"],
        port=settings["DISCOVER_PORT"],
        secure=settings["SECURE"],
        verify=settings["VERIFY"],
        token=settings["ADMIN_TOKEN"],
    )

    result = client.set_active_model(seed_path)
    print(result["message"])


if __name__ == "__main__":
    @click.command()
    @click.argument("seed_path", type=str, default="seed.npz")
    def main(seed_path):
        """Initialize FEDn with a seed model from the specified path."""
        init_fedn(seed_path)

    main()
