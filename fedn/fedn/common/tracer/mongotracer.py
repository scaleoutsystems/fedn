
from fedn.common.tracer.tracer import Tracer
from fedn.common.storage.db.mongo import connect_to_mongodb


class MongoTracer(Tracer):
    def __init__(self):
        try:
            self.mdb = connect_to_mongodb()
            self.collection = self.mdb['status']
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.collection = None
            raise

    def report(self, msg):
        from google.protobuf.json_format import MessageToDict
        data = MessageToDict(msg, including_default_value_fields=True)

        print("LOG: \n {} \n".format(data),flush=True)

        if self.collection:
            self.collection.insert_one(data)
