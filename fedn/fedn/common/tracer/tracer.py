from abc import ABC, abstractmethod


class Tracer(ABC):
    @abstractmethod
    def report_status(self, msg):
        """

        :param msg:
        """
        pass
