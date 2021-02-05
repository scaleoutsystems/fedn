from .certificate import Certificate


class CertificateManager:

    def __init__(self, directory):

        self.directory = directory
        self.certificates = []
        self.allowed = dict()
        self.load_all()

    def get_or_create(self, name):
        search = self.find(name)
        if search:
            return search
        else:
            cert = Certificate(self.directory, name=name, cert_name=name + '-cert.pem', key_name=name + '-key.pem')
            cert.gen_keypair()
            self.certificates.append(cert)
            return cert

    def add(self, certificate):
        if not self.find(certificate.name):
            self.certificates.append(certificate)
            return True
        return False

    def load_all(self):
        import os
        for filename in sorted(os.listdir(self.directory)):
            if filename.endswith('cert.pem'):
                name = filename.split('-')[0]
                #print("got a file here! Read it {}".format(filename))
                key_name = name + '-key.pem'
                #print("trying with {}".format(key_name))
                if os.path.isfile(os.path.join(self.directory,key_name)):

                    c = Certificate(self.directory, name=name, cert_name=filename, key_name=key_name)
                    self.certificates.append(c)
                else:
                    c = Certificate(self.directory, name=name, cert_name=filename, key_name=key_name)  # , cert_only=True)
                    self.certificates.append(c)

    def find(self, name):
        for cert in self.certificates:
            if cert.name == name:
                return cert
        for cert in self.allowed:
            if cert.name == name:
                return cert
        return None
