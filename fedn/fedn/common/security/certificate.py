import os

from OpenSSL import crypto


class Certificate:
    CERT_NAME = "cert.pem"
    KEY_NAME = "key.pem"
    BITS = 2048

    def __init__(self, cwd, name=None, key_name="key.pem", cert_name="cert.pem", create_dirs=True):

        try:
            os.makedirs(cwd)
        except OSError:
            print("Directory exists, will store all cert and keys here.")
        else:
            print("Successfully created the directory to store cert and keys in {}".format(cwd))
        self.key_path = os.path.join(cwd, key_name)
        self.cert_path = os.path.join(cwd, cert_name)
        import uuid
        if name:
            self.name = name
        else:
            self.name = str(uuid.uuid4())

    def gen_keypair(self, ):
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)
        cert = crypto.X509()
        cert.get_subject().C = "SE"
        cert.get_subject().ST = "Stockholm"
        cert.get_subject().O = "Development Key"
        cert.get_subject().OU = "Development Key"
        cert.get_subject().CN = self.name  # gethostname()

        import random
        cert.set_serial_number(int(random.randint(1000,100000)))

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
        with open(self.key_path, "wb") as keyfile:
            keyfile.write(privatekey)

        with open(self.cert_path, "wb") as certfile:
            certfile.write(certificate)

    def get_keypair_raw(self):
        with open(self.key_path, 'rb') as keyfile:
            key_buf = keyfile.read()
        with open(self.cert_path, 'rb') as certfile:
            cert_buf = certfile.read()
        import copy
        return copy.deepcopy(cert_buf), copy.deepcopy(key_buf)

    def get_key(self):
        with open(self.key_path, 'rb') as keyfile:
            key_buf = keyfile.read()
        key = crypto.load_privatekey(crypto.FILETYPE_PEM, key_buf)
        return key

    def get_cert(self):
        with open(self.cert_path, 'rb') as certfile:
            cert_buf = certfile.read()
        cert = crypto.load_certificate(crypto.FILETYPE_PEM, cert_buf)
        return cert

    def __str__(self):
        return "Certificate name: {}".format(self.name)
