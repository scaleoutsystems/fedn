from abc import ABC, abstractmethod


class Tracer(ABC):
    @abstractmethod
    def report(self, msg):
        pass
