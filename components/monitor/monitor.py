import threading
import os
import time
import grpc
import requests
import json

from google.protobuf.json_format import MessageToJson, MessageToDict

import fedn.proto.alliance_pb2 as alliance
import fedn.proto.alliance_pb2_grpc as rpc

import pymongo
from fedn.utils.mongo import connect_to_mongodb


class Client:
    def __init__(self, url, port, name):

        self.name = name
        channel = grpc.insecure_channel(url + ":" + str(port))
        self.connection = rpc.ConnectorStub(channel)
        self.headers = {'Content-Type': 'application/json'}
        print("Client: {} connected to {}:{}".format(self.name, url, port), flush=True)

        # Alliance admin enpoints
        self.api_url = None
        try:
            self.api_url = str.format('{}/projects/{}/{}/alliance_admin/{}/log',
                                      os.environ['API_URL'], os.environ['USER'], os.environ['PROJECT'],
                                      os.environ['ALLIANCE_UID'])
        except Exception:
            pass

        # Connect to MongoDB 
        try:
            # TODO: Get configs from env
            self.mc = pymongo.MongoClient('mongo', 27017, username='root', password='example')
            # This is so that we check that the connection is live
            self.mc.server_info()
            self.mdb = self.mc[os.environ['ALLIANCE_UID']]
            self.collection = self.mdb['status']
        except Exception:
            self.collection = None
            pass

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

            if self.api_url:
                r = requests.post(self.api_url, data={'message': str(data)})

    def run(self):
        print("starting")
        while True:
            print(".")
            time.sleep(1)


def main():
    url = os.environ['MONITOR_HOST']
    port = os.environ['MONITOR_PORT']
    if url is None or port is None:
        print("Cannot start without MONITOR_HOST and MONITOR_PORT")
        return

    c = Client(url, port, "monitor")
    c.run()


if __name__ == '__main__':
    print("MONITOR: starting up", flush=True)
    main()
