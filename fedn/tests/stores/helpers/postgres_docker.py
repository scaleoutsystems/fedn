import docker
import time
import os

CONTAINER_NAME="postgres-test-db"
""" 
    An alternative to this is to create the container using
    githiub action and then use the container to create the client
"""
def start_postgres_container():
    client = docker.from_env()        
    already_running = False
    try:
        container = client.containers.get(CONTAINER_NAME)
        already_running = True
    except docker.errors.NotFound:
        container = client.containers.run(
            "postgres:15",
            detach=True,
            ports={"5432/tcp": None}, # Let Docker choose an available port
            name=CONTAINER_NAME,
            environment={
                "POSTGRES_USER": os.environ.get("TEST_USER", "_"),
                "POSTGRES_PASSWORD": os.environ.get("TEST_PASS", "_"),
                "POSTGRES_DB": "fedn_db"
            }
        )
    time.sleep(1)
    start = time.time()
    while time.time() - start < 10:
        try:
            container.reload()
            port = int(container.attrs['NetworkSettings']['Ports']['5432/tcp'][0]['HostPort'])
            break
        except:
            time.sleep(1)
    else:
        raise Exception("Could not start Postgres container")

    return already_running, container, port

def stop_postgres_container():
    client = docker.from_env()
    container = client.containers.get(CONTAINER_NAME)
    container.stop()
    container.remove(v=True, force=True)
    try:
        container.wait(condition="removed")
    except docker.errors.NotFound:
        pass


