from config import settings
from fedn import APIClient

client = APIClient(
    host=settings["DISCOVER_HOST"],
    port=None, #settings["DISCOVER_PORT"],
    secure=settings["SECURE"],
    verify=settings["VERIFY"],
    token=settings["ADMIN_TOKEN"],
)

result = client.set_active_model("seed.npz")
print(result["message"])
