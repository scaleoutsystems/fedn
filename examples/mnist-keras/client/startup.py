import train
import validate


def startup(client):
    global client_reference
    client_reference = client
    client.set_train_callback(train.train)
    client.set_validate_callback(validate.validate)
