import uuid
from datetime import datetime

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
            self.validations = self.mdb['control.validations']
            self.clients = self.mdb['network.clients']
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.status = None
            raise

    def report_status(self, msg):
        """Write status message to the database.

        :param msg: The status message.
        """
        data = MessageToDict(msg, including_default_value_fields=True)

        if self.status is not None:
            self.status.insert_one(data)

    def report_validation(self, validation):
        """Write model validation to the database.

        :param validation: The model validation.
        """
        data = MessageToDict(validation, including_default_value_fields=True)

        if self.validations is not None:
            self.validations.insert_one(data)

    def drop_status(self):
        """Drop the status collection.

        """
        if self.status:
            self.status.drop()

    def create_session(self, id=None):
        """ Create a new session.

        :param id: The ID of the created session.
        :type id: uuid, str

        """
        if not id:
            id = uuid.uuid4()
        data = {'session_id': str(id)}
        self.sessions.insert_one(data)

    def create_round(self, round_data):
        """ Create a new round.

        :param round_data: Dictionary with round data.
        :type round_data: dict
        """
        # TODO: Add check if round_id already exists
        self.rounds.insert_one(round_data)

    def set_session_config(self, id, config):
        self.sessions.update_one({'session_id': str(id)}, {
            '$push': {'session_config': config}}, True)

    def set_round_combiner_data(self, data):
        """

        :param data: The combiner data
        :type data: dict
        """
        self.rounds.update_one({'round_id': str(data['round_id'])}, {
            '$push': {'combiners': data}}, True)

    def set_round_config(self, round_id, round_config):
        """

        :param round_meta:
        """
        self.rounds.update_one({'round_id': round_id}, {
            '$set': {'round_config': round_config}}, True)

    def set_round_status(self, round_id, round_status):
        """

        :param round_meta:
        """
        self.rounds.update_one({'round_id': round_id}, {
            '$set': {'status': round_status}}, True)

    def set_round_data(self, round_id, round_data):
        """

        :param round_meta:
        """
        self.rounds.update_one({'round_id': round_id}, {
            '$set': {'round_data': round_data}}, True)

    def update_client_status(self, client_name, status):
        """ Update client status in statestore.
        :param client_name: The client name
        :type client_name: str
        :param status: The client status
        :type status: str
        :return: None
        """
        datetime_now = datetime.now()
        filter_query = {"name": client_name}

        update_query = {"$set": {"last_seen": datetime_now, "status": status}}
        self.clients.update_one(filter_query, update_query)
