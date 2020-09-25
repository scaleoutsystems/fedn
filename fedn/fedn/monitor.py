import threading
import time

import fedn.common.net.grpc.fedn_pb2 as alliance
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
import grpc
from fedn.common.storage.db.mongo import connect_to_mongodb
from google.protobuf.json_format import MessageToDict


class Monitor:
    def __init__(self, config):

        if config['secure']:
            raise Exception("NYI - Monitor does not support secure mode yet. Use builtin monitor in combiner mode")
        self.name = config['name']
        channel = grpc.insecure_channel(config['host'] + ":" + str(config['port']))
        self.connection = rpc.ConnectorStub(channel)
        print("Client: {} connected to {}:{}".format(self.name, config['host'], config['port']), flush=True)

        # Connect to MongoDB
        try:
            self.mdb = connect_to_mongodb()
            self.collection = self.mdb['status']
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.collection = None
            raise

        threading.Thread(target=self.__listen_to_status_stream, daemon=True).start()

    def __listen_to_status_stream(self):
        r = alliance.ClientAvailableMessage()
        r.sender.name = self.name
        r.sender.role = alliance.OTHER
        for status in self.connection.AllianceStatusStream(r):
            print("MONITOR: Recived status:{}".format(status), flush=True)
            data = MessageToDict(status, including_default_value_fields=True)
            print(data, flush=True)

            # Log to MongoDB
            if self.collection:
                self.collection.insert_one(data)

    def run(self):

        print("starting")
        while True:
            print(".")
            time.sleep(1)
