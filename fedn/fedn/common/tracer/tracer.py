from abc import ABC, abstractmethod


class Tracer(ABC):
    @abstractmethod
    def report(self, msg):
        """

        :param msg:
        """
        pass
