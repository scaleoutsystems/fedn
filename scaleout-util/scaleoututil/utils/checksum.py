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


def compute_checksum_from_stream(stream):
    """Compute the SHA256 checksum from a stream.

    :param stream: The stream to compute the checksum from.
    :type stream: io.BytesIO or similar
    :return: The SHA256 checksum as a string.
    :rtype: str
    """
    hash = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4096), b""):
        hash.update(chunk)
    return hash.hexdigest()


def compute_checksum_from_file(file_path):
    """Compute the SHA256 checksum from a file.

    :param file_path: The path to the file.
    :type file_path: str
    :return: The SHA256 checksum as a string.
    :rtype: str
    """
    with open(file_path, "rb") as f:
        return compute_checksum_from_stream(f)
