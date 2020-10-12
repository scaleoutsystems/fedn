
from fedn.common.tracer.tracer import Tracer
from fedn.common.storage.db.mongo import connect_to_mongodb
import time
import os
import psutil

class MongoTracer(Tracer):
    def __init__(self):
        try:
            self.mdb = connect_to_mongodb()
            self.collection = self.mdb['status']
            self.performances = self.mdb['performances']
            self.psutil_usage = self.mdb['psutil_usage']
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

    def drop_performances(self):
        if self.performances:
            self.performances.drop()

    def set_latest_time(self, round, round_time):
        self.performances.update({'key': 'performance'}, {'$push': {'round': round}}, True)
        self.performances.update({'key': 'performance'}, {'$push': {'time': round_time}}, True)

    def ps_util_monitor(self, target):
        import multiprocessing as mp
        worker_process = mp.Process(target=target)
        worker_process.start()
        p = psutil.Process(worker_process.pid)

        # log cpu usage of `worker_process` every 10 ms
        cpu_percents = []
        mem_percents = []
        ps_time = []
        start_time = 0
        while worker_process.is_alive():
            cpu_percents.append(p.cpu_percent())
            # (Resident Set Size) memory that a task has used (in kiloBytes)
            # mem_percents.append(p.memory_info().rss / float(2 ** 20))
            mem_percents.append(p.memory_percent())
            ps_time.append(start_time)
            start_time += 10
            time.sleep(0.01)

        worker_process.join()
        print('----CPU----', cpu_percents)
        print('----MEM----', mem_percents)
        if self.psutil_usage:
            self.psutil_usage.drop()
        self.psutil_usage.update({'key': 'cpu_mem_usage'}, {'$push': {'cpu': {'$each': cpu_percents}}}, True)
        self.psutil_usage.update({'key': 'cpu_mem_usage'}, {'$push': {'mem': {'$each': mem_percents}}}, True)
        self.psutil_usage.update({'key': 'cpu_mem_usage'}, {'$push': {'time': {'$each': ps_time}}}, True)
        return cpu_percents, mem_percents