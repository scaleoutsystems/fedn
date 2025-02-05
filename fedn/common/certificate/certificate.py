import copy
import os
import random
import uuid

from OpenSSL import crypto

from fedn.common.log_config import logger


class Certificate:
    """Utility to generate unsigned certificates."""

    CERT_NAME = "cert.pem"
    KEY_NAME = "key.pem"
    BITS = 2048

    def __init__(self, name=None, key_path="", cert_path="", create_dirs=False):
        if create_dirs:
            try:
                cwd = os.getcwd()
                os.makedirs(cwd)
            except OSError:
                logger.info("Directory exists, will store all cert and keys here.")
            else:
                logger.info("Successfully created the directory to store cert and keys in {}".format(cwd))

            self.key_path = os.path.join(cwd, "key.pem")
            self.cert_path = os.path.join(cwd, "cert.pem")
        else:
            self.key_path = key_path
            self.cert_path = cert_path

        if name:
            self.name = name
        else:
            self.name = str(uuid.uuid4())

    def gen_keypair(
        self,
    ):
        """Generate keypair."""
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)
        cert = crypto.X509()
        cert.get_subject().C = "SE"
        cert.get_subject().ST = "Stockholm"
        cert.get_subject().O = "Development Key"  # noqa: E741
        cert.get_subject().OU = "Development Key"
        cert.get_subject().CN = self.name

        cert.set_serial_number(int(random.randint(1000, 100000)))

        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(31 * 24 * 60 * 60)

        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, "sha256")
        with open(self.key_path, "wb") as keyfile:
            keyfile.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

        with open(self.cert_path, "wb") as certfile:
            certfile.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))

    def set_keypair_raw(self, certificate, privatekey):
        """:param certificate:
        :param privatekey:
        """
        with open(self.key_path, "wb") as keyfile:
            keyfile.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, privatekey))

        with open(self.cert_path, "wb") as certfile:
            certfile.write(crypto.dump_certificate(crypto.FILETYPE_PEM, certificate))

    def get_keypair_raw(self):
        """:return:"""
        with open(self.key_path, "rb") as keyfile:
            key_buf = keyfile.read()
        with open(self.cert_path, "rb") as certfile:
            cert_buf = certfile.read()
        return copy.deepcopy(cert_buf), copy.deepcopy(key_buf)

    def get_key(self):
        """:return:"""
        with open(self.key_path, "rb") as keyfile:
            key_buf = keyfile.read()
        key = crypto.load_privatekey(crypto.FILETYPE_PEM, key_buf)
        return key

    def get_cert(self):
        """:return:"""
        with open(self.cert_path, "rb") as certfile:
            cert_buf = certfile.read()
        cert = crypto.load_certificate(crypto.FILETYPE_PEM, cert_buf)
        return cert

    def __str__(self):
        return "Certificate name: {}".format(self.name)
