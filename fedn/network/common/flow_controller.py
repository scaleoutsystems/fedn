import threading
import time
from enum import Enum


class FlowController:
    class Reason(Enum):
        """Reason for the flow controller resuming."""

        STOP = "stop"
        CONTINUE = "continue"
        TIMEOUT = "timeout"
        CONDITION = "condition"

    def __init__(self):
        self.stop_event = threading.Event()
        self.continue_event = threading.Event()

    def wait_until_true(self, callback, timeout=0.0, polling_rate=1.0) -> Reason:
        """Wait until the callback returns True or the timeout is reached.

        :param callback: The callback function to call.
        :type callback: function
        :param timeout: The timeout in seconds, defaults to 0.0 which means no timeout.
        :type timeout: float, optional
        :param polling_rate: The polling rate in seconds, defaults to 1.0.
        :type polling_rate: float, optional
        :return: The reason for the flow controller resuming.
        :rtype: Reason
        """
        self.continue_event.clear()
        start = time.time()

        while True:
            if callback():
                return self.Reason.CONDITION
            if self.continue_event.is_set():
                return self.Reason.CONTINUE
            if self.stop_event.is_set():
                return self.Reason.STOP
            if timeout > 0.0 and time.time() - start > timeout:
                return self.Reason.TIMEOUT
            time.sleep(polling_rate)
