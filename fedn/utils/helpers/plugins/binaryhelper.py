from fedn.utils.helpers.plugins.numpyhelper import Helper


class Helper(Helper):
    """FEDn helper class for models weights/parameters that can be transformed to numpy ndarrays."""

    def __init__(self):
        """Initialize helper."""
        super().__init__()
        self.name = "binaryhelper"

    def load(self, path, file_type="raw_binary"):
        return super().load(path, file_type)

    def save(self, model, path=None, file_type="raw_binary"):
        return super().save(model, path, file_type)
