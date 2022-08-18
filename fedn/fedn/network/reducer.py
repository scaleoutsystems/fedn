import os
import re
import threading
import time
from datetime import datetime

from fedn.common.security.certificatemanager import CertificateManager
from fedn.network.controller.control import Control
from fedn.network.dashboard.restservice import ReducerRestService
from fedn.network.state import ReducerStateToString

VALID_NAME_REGEX = '^[a-zA-Z0-9_-]*$'


class InvalidReducerConfiguration(Exception):
    pass


class MissingReducerConfiguration(Exception):
    pass


class Reducer:
    """ A class used to instantiate the Reducer service.

    Start Reducer services.
    """

    def __init__(self, statestore):
        """
        Parameters
        ----------
        statestore: dict
            The backend statestore object.
        """

        self.statestore = statestore
        config = self.statestore.get_reducer()
        if not config:
            print("REDUCER: Failed to retrive Reducer config, exiting.")
            raise MissingReducerConfiguration()

        print(config, flush=True)
        # Validate reducer name
        match = re.search(VALID_NAME_REGEX, config['name'])
        if not match:
            raise ValueError('Unallowed character in reducer name. Allowed characters: a-z, A-Z, 0-9, _, -.')
        self.name = config['name']

        # The certificate manager is a utility that generates (self-signed) certificates.
        self.certificate_manager = CertificateManager(os.getcwd() + "/certs/")

        self.control = Control(self.statestore)

        self.rest = ReducerRestService(
            config, self.control, self.certificate_manager)

    def run(self):
        """Start REST service and control loop."""

        threading.Thread(target=self.control_loop, daemon=True).start()

        self.rest.run()

    def control_loop(self):
        """Manage and report the state of the Reducer."""

        try:
            old_state = self.control.state()

            t1 = datetime.now()
            while True:
                time.sleep(1)
                if old_state != self.control.state():
                    delta = datetime.now() - t1
                    print(
                        "Reducer in state {} for {} seconds. Entering {} state".format(ReducerStateToString(old_state),
                                                                                       delta.seconds,
                                                                                       ReducerStateToString(
                                                                                           self.control.state())),
                        flush=True)
                    t1 = datetime.now()
                    old_state = self.control.state()

                self.control.monitor()
        except (KeyboardInterrupt, SystemExit):
            print("Exiting..", flush=True)
