import uuid

from google.protobuf.json_format import MessageToDict

from fedn.common.storage.db.mongo import connect_to_mongodb
from fedn.common.tracer.tracer import Tracer


class MongoTracer(Tracer):
    """ Utitily for reporting and tracking state in the statestore.

    """

    def __init__(self, mongo_config, network_id):
        try:
            self.mdb = connect_to_mongodb(mongo_config, network_id)
            self.status = self.mdb['control.status']
            self.rounds = self.mdb['control.rounds']
            self.sessions = self.mdb['control.sessions']
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.status = None
            raise

    def report(self, msg):
        """

        :param msg:
        """
        data = MessageToDict(msg, including_default_value_fields=True)

        if self.status is not None:
            self.status.insert_one(data)

    def drop_status(self):
        """

        """
        if self.status:
            self.status.drop()

    def new_session(self, id=None):
        """ Create a new session. """
        if not id:
            id = uuid.uuid4()
        data = {'session_id': str(id)}
        self.sessions.insert_one(data)

    def set_session_config(self, id, config):
        self.sessions.update_one({'session_id': str(id)}, {
            '$push': {'session_config': config}}, True)

    def set_round_meta(self, round_meta):
        """

        :param round_meta:
        """
        self.rounds.update_one({'key': str(round_meta['round_id'])}, {
            '$push': {'combiners': round_meta}}, True)

    def set_round_meta_reducer(self, round_meta):
        """

        :param round_meta:
        """
        self.rounds.update_one({'key': str(round_meta['round_id'])}, {
            '$push': {'reducer': round_meta}}, True)

    def get_latest_round(self):
        """

        :return:
        """
        for post in self.round_time.find({'key': 'round_time'}):
            last_round = post['round'][-1]
            return last_round
