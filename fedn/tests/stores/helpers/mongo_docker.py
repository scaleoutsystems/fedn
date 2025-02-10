import docker
import time

CONTAINER_NAME="mongo-test-db"
""" 
    An alternative to this is to create the MongoDB container using
    githiub action and then use the container to create the client
"""
def start_mongodb_container():
    client = docker.from_env()        

    already_running = False
    try:
        container = client.containers.get(CONTAINER_NAME)
        already_running = True
    except docker.errors.NotFound:
        container = client.containers.run(
            "mongo:7.0",
            detach=True,
            ports={"27017/tcp": None}, # Let Docker choose an available port
            name=CONTAINER_NAME,
            environment={
                "MONGO_INITDB_ROOT_USERNAME": "fedn_admin",
                "MONGO_INITDB_ROOT_PASSWORD": "password"
            },
            command="mongod"
        )
        
    time.sleep(1)
    start = time.time()
    while time.time() - start < 10:
        try:
            container.reload()
            port = int(container.attrs['NetworkSettings']['Ports']['27017/tcp'][0]['HostPort'])
            break
        except:
            time.sleep(1)
    else:
        raise Exception("Could not start MongoDB container")

    return already_running, container, port

def stop_mongodb_container():
    client = docker.from_env()
    container = client.containers.get(CONTAINER_NAME)
    container.stop()
    container.remove(v=True, force=True)
    try:
        container.wait(condition="removed")
    except docker.errors.NotFound:
        pass


