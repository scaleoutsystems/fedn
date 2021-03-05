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
            self.status = self.mdb['control.status']
            self.round_time = self.mdb['control.round_time']
            self.psutil_monitoring = self.mdb['control.psutil_monitoring']
            self.model_trail = self.mdb['control.model_trail']
            self.latest_model = self.mdb['control.latest_model']
            self.combiner_round_time = self.mdb['control.combiner_round_time']
            #self.combiner_queue_length = self.mdb['control.combiner_queue_length']
            self.round = self.mdb['control.round']
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.status = None
            raise

    def report(self, msg):
        from google.protobuf.json_format import MessageToDict
        data = MessageToDict(msg, including_default_value_fields=True)

        print("LOG: \n {} \n".format(data),flush=True)

        if self.status:
            self.status.insert_one(data)

    def drop_round_time(self):
        if self.round_time:
            self.round_time.drop()

    def drop_ps_util_monitor(self):
        if self.psutil_monitoring:
            self.psutil_monitoring.drop()

    def drop_model_trail(self):
        if self.model_trail:
            self.model_trail.drop()

    def drop_latest_model(self):
        if self.latest_model:
            self.latest_model.drop()

    def drop_status(self):
        if self.status:
            self.status.drop()

    def drop_combiner_round_time(self):
        if self.combiner_round_time:
            self.combiner_round_time.drop()

    def drop_combiner_round(self):
        if self.round:
            self.round.drop()

    def set_latest_time(self, round, round_time):
        self.round_time.update({'key': 'round_time'}, {'$push': {'round': round}}, True)
        self.round_time.update({'key': 'round_time'}, {'$push': {'round_time': round_time}}, True)

    def set_combiner_time(self, round, round_time):
        self.combiner_round_time.update({'key': 'combiner_round_time'}, {'$push': {'round': round}}, True)
        self.combiner_round_time.update({'key': 'combiner_round_time'}, {'$push': {'round_time': round_time}}, True)

    #def set_combiner_queue_length(self,timestamp,ql):
    #    self.combiner_queue_length({'key': 'combiner_queue_length'}, {'$push': {'queue_length': ql}}, True)
    #    self.combiner_queue_length.update({'key': 'combiner_queue_length'}, {'$push': {'timestamp': timestamp}}, True)

    # Round statistics
    def set_round_meta(self, round_meta):
        self.round.update({'key': str(round_meta['round_id'])}, {'$push': {'combiners': round_meta}}, True)
    
    def set_round_meta_reducer(self, round_meta):
        self.round.update({'key': str(round_meta['round_id'])}, {'$push': {'reducer': round_meta}}, True)

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