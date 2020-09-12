import threading

from fedn.clients.reducer.control import ReducerControl
from fedn.clients.reducer.interfaces import ReducerInferenceInterface
from fedn.clients.reducer.restservice import ReducerRestService
from fedn.clients.reducer.state import ReducerStateToString


class Reducer:
    def __init__(self, config):
        self.name = config['name']
        self.token = config['token']
        self.control = ReducerControl()
        self.inference = ReducerInferenceInterface()
        self.rest = ReducerRestService(config['name'], self.control)

    def run(self):

        threading.Thread(target=self.rest.run, daemon=True).start()

        import time
        try:
            while True:
                time.sleep(1)
                print("Reducer in {} state".format(ReducerStateToString(self.control.state())), flush=True)
                self.control.monitor()
        except (KeyboardInterrupt, SystemExit):
            print("Exiting..", flush=True)
