import os
import threading

from fedn.clients.reducer.control import ReducerControl
from fedn.clients.reducer.interfaces import ReducerInferenceInterface
from fedn.clients.reducer.restservice import ReducerRestService
from fedn.clients.reducer.state import ReducerStateToString
from fedn.common.security.certificatemanager import CertificateManager
from fedn.clients.reducer.statestore.mongoreducerstatestore import MongoReducerStateStore

class InvalidReducerConfiguration(Exception):
    pass

class MissingReducerConfiguration(Exception):
    pass

class Reducer:
    def __init__(self, statestore):
        """ """ 
        self.statestore = statestore
        config = self.statestore.get_reducer()
        if not config:
            print("REDUCER: Failed to retrive Reducer config, exiting.")
            raise MissingReducerConfiguration()

        self.name = config['name']
        self.token = config['token']
        
        try:
            path = config['path']
        except KeyError:
            path = os.getcwd()

        self.certificate_manager = CertificateManager(os.getcwd() + "/certs/")

        self.control = ReducerControl(self.statestore)
        self.inference = ReducerInferenceInterface()
        rest_certificate = self.certificate_manager.get_or_create("reducer")
        self.rest = ReducerRestService(config, self.control, self.certificate_manager, certificate=rest_certificate)

    def run(self):

        threading.Thread(target=self.control_loop, daemon=True).start()

        self.rest.run()

    def control_loop(self):
        import time
        from datetime import datetime
        try:
            old_state = self.control.state()

            t1 = datetime.now()
            while True:
                time.sleep(1)
                if old_state != self.control.state():
                    delta = datetime.now() - t1
                    print("Reducer in state {} for {} seconds. Entering {} state".format(ReducerStateToString(old_state),delta.seconds,ReducerStateToString(self.control.state())), flush=True)
                    t1 = datetime.now()
                    old_state = self.control.state()

                self.control.monitor()
        except (KeyboardInterrupt, SystemExit):
            print("Exiting..", flush=True)