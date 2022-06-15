import hashlib


def sha(fname):
    """

    :param fname:
    :return:
    """
    hash = hashlib.SHA256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash.hexdigest()
