import os

from .certificate import Certificate


class CertificateManager:
    """Utility to handle certificates for both Reducer and Combiner services.

    """

    def __init__(self, directory):
        self.directory = directory
        self.certificates = []
        self.allowed = dict()
        self.load_all()

    def get_or_create(self, name):
        """Look for an existing certificate, if not found, generate a self-signed certificate based on name.

        :param name: The name used when issuing the certificate.
        :return: A certificate
        :rtype: str
        """
        search = self.find(name)
        if search:
            return search
        else:
            cert = Certificate(self.directory, name=name, cert_name=name + "-cert.pem", key_name=name + "-key.pem")
            cert.gen_keypair()
            self.certificates.append(cert)
            return cert

    def add(self, certificate):
        """Add certificate to certificate list.

        :param certificate:
        :return: Success status (True, False)
        :rtype: Boolean
        """
        if not self.find(certificate.name):
            self.certificates.append(certificate)
            return True
        return False

    def load_all(self):
        """Load all certificates and add to certificates list.

        """
        for filename in sorted(os.listdir(self.directory)):
            if filename.endswith("cert.pem"):
                name = filename.split("-")[0]
                key_name = name + "-key.pem"

                c = Certificate(self.directory, name=name, cert_name=filename, key_name=key_name)
                self.certificates.append(c)

    def find(self, name):
        """:param name: Name of certificate
        :return: certificate if successful, else None
        """
        for cert in self.certificates:
            if cert.name == name:
                return cert
        for cert in self.allowed:
            if cert.name == name:
                return cert
        return None
