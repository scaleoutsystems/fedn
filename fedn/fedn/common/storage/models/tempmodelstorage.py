
from .modelstorage import ModelStorage


def TempModelStorage(ModelStorage):

    def __init__(self):
        import tempfile
        self.dir = tempfile.TemporaryDirectory()