from fedn.common.tracer.tracer import Tracer
from fedn.common.storage.db.mongo import connect_to_mongodb
import time
import threading
import psutil
from datetime import datetime

class MongoTracer(Tracer):
    def __init__(self,mongo_config,network_id):
        try:
            self.mdb = connect_to_mongodb(mongo_config,network_id)
            self.collection = self.mdb['status']
            self.round_time = self.mdb['round_time']
            self.psutil_monitoring = self.mdb['psutil_monitoring']
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

    def drop_round_time(self):
        if self.round_time:
            self.round_time.drop()

    def drop_ps_util_monitor(self):
        if self.psutil_monitoring:
            self.psutil_monitoring.drop()

    def set_latest_time(self, round, round_time):
        self.round_time.update({'key': 'round_time'}, {'$push': {'round': round}}, True)
        self.round_time.update({'key': 'round_time'}, {'$push': {'round_time': round_time}}, True)

    def get_latest_round(self):
        for post in self.round_time.find({'key': 'round_time'}):
            last_round = post['round'][-1]
            return last_round

    def ps_util_monitor(self, round=None):
        global running
        running = True
        currentProcess = psutil.Process()
        # start loop
        while running:
            cpu_percents = currentProcess.cpu_percent(interval=1)
            mem_percents = currentProcess.memory_percent()
            ps_time = str(datetime.now())

            self.psutil_monitoring.update({'key': 'cpu_mem_usage'}, {'$push': {'cpu': cpu_percents}}, True)
            self.psutil_monitoring.update({'key': 'cpu_mem_usage'}, {'$push': {'mem': mem_percents}}, True)
            self.psutil_monitoring.update({'key': 'cpu_mem_usage'}, {'$push': {'time': ps_time}}, True)
            self.psutil_monitoring.update({'key': 'cpu_mem_usage'}, {'$push': {'round': round}}, True)

    def start_monitor(self, round=None):
        global t
        # create thread and start it
        t = threading.Thread(target=self.ps_util_monitor, args=[round])
        t.start()

    def stop_monitor(self):
        global running
        global t
        # use `running` to stop loop in thread so thread will end
        running = False
        # wait for thread's end
        t.join()