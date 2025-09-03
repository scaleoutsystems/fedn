import time
import uuid

from config import settings
from fedn import APIClient

client = APIClient(
    host= settings["DISCOVER_HOST"],
    port= None,#settings["DISCOVER_PORT"],
    secure=settings["SECURE"],
    verify=settings["VERIFY"],
    token=settings["ADMIN_TOKEN"],
)

if __name__ == "__main__":
    # Run six sessions, each with 100 rounds.
    for s in range(settings["N_SESSIONS"]):
        active_model = client.get_active_model()
        print(active_model)
        session_config = {
            "helper": "numpyhelper",
            "name": f"async-test-{s+1}-{str(uuid.uuid4())[:4]}",
            "aggregator": "fedavg",
            "round_timeout": settings["ROUND_TIMEOUT"],
            "rounds": settings["N_ROUNDS"],
            "validate": True,
            "model_id": active_model["model"],
        }

        session = client.start_session(**session_config)
        if session["message"] != "Session started":
            print(session["message"])
            exit(0)

        print("Started session: {}".format(session))
        session_id = session["session_id"]

        # Wait for session to finish
        while not client.session_is_finished(session_id):
            t_sleep = 2
            session_status = client.get_session_status(session_id)
            print(f"Session status: {session_status}. Sleeping for {t_sleep}")
            time.sleep(t_sleep)