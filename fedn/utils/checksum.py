import hashlib


def sha(fname):
    """Calculate the sha256 checksum of a file. Used for computing checksums of compute packages.

    :param fname: The file path.
    :type fname: str
    :return: The sha256 checksum.
    :rtype: :py:class:`hashlib.sha256`
    """
    hash = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash.hexdigest()
