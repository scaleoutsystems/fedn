import os

from fedn.utils.dispatcher import Dispatcher


class Package:

    def __init__(self, config):
        self.config = config
        self.name = config['name']
        self.cwd = config['cwd']
        if 'port' in config:
            self.reducer_port = config['port']
        if 'host' in config:
            self.reducer_host = config['host']
        if 'token' in config:
            self.reducer_token = config['token']

        self.package_file = None
        self.file_path = None
        self.package_hash = None

    def package(self, validate=False):

        # check config
        package_file = '{name}.tar.gz'.format(name=self.name)

        # package the file
        import os
        cwd = os.getcwd()
        self.file_path = os.getcwd()
        if self.config['cwd'] == '':
            self.file_path = os.getcwd()
        os.chdir(self.file_path)

        import tarfile
        with tarfile.open(os.path.join(os.path.dirname(self.file_path), package_file), 'w:gz') as tf:
            # for walking the current dir with absolute path (in archive)
            #for root, dirs, files in os.walk(self.file_path):
                #for file in files:
                    #tf.add(os.path.join(root, file))
            # for walking the current dir
            for file in os.listdir(self.file_path):
                tf.add(file)
            tf.close()

        import hashlib
        hsh = hashlib.sha256()
        with open(os.path.join(os.path.dirname(self.file_path), package_file), 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                hsh.update(byte_block)

        os.chdir(cwd)
        self.package_file = package_file
        self.package_hash = hsh.hexdigest()

        return package_file, hsh.hexdigest()

    def upload(self):
        if self.package_file:
            import requests
            import os
            print("Upload 1 ",flush=True)
            #data = {'name': self.package_file, 'hash': str(self.package_hash)}
            #print("going to send {}".format(data),flush=True)
            f = open(os.path.join(os.path.dirname(self.file_path), self.package_file), 'rb')
            print("Sending the following file {}".format(f.read()), flush=True)
            f.seek(0,0)
            print("Upload 2 ", flush=True)
            files = {'file': f}
            try:
                retval = requests.post('https://{}:{}/context'.format(self.reducer_host, self.reducer_port), verify=False, files=files,
                              #data=data,
                              headers={'Authorization': 'Token {}'.format(self.reducer_token)})
                print("Upload 3 Returned:{}".format(retval), flush=True)
            except Exception as e:
                print("failed to put execution context to reducer. {}".format(e), flush=True)
            finally:
                f.close()

            print("Upload 4 ", flush=True)


class PackageRuntime:

    def __init__(self, package_path, package_dir):

        self.dispatch_config = {'entry_points':
                                    {'predict': {'command': 'python3 predict.py'},
                                     'train': {'command': 'python3 train.py'},
                                     'validate': {'command': 'python3 validate.py'}}}

        self.pkg_path = package_path
        self.pkg_name = None
        self.dir = package_dir

    def download(self, host, port, token, name=None):
        import requests
        print("Download 1 ",flush=True)
        path = "https://{}:{}/context".format(host, port)
        if name:
            path = path + "?name={}".format(name)
        print("Download 2 ", flush=True)
        with requests.get(path, stream=True, verify=False, headers={'Authorization': 'Token {}'.format(token)}) as r:
            import cgi
            params = cgi.parse_header(r.headers.get('Content-Disposition', ''))[-1]
            self.pkg_name = params['filename']
            print("Download 3 ", flush=True)

            r.raise_for_status()
            with open(os.path.join(self.pkg_path, self.pkg_name), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download 4 ", flush=True)

    def validate(self):
        # crosscheck checksum and unpack if security checks are ok.
        pass

    def unpack(self):
        import os
        import tarfile
        print("Unpack 1 ", flush=True)
        f = None
        if self.pkg_name.endswith('tar.gz'):
            f = tarfile.open(os.path.join(self.pkg_path, self.pkg_name), 'r:gz')
        if self.pkg_name.endswith('tar.bz2'):
            f = tarfile.open(os.path.join(self.pkg_path, self.pkg_name), 'r:bz2')
        import os
        cwd = os.getcwd()
        try:
            print("trying to create dirs {}".format(self.dir),flush=True)
            #os.makedirs(self.dir)
            print("trying to change to dir {}".format(self.dir),flush=True)
            os.chdir(self.dir)
            print("Unpack 2 ", flush=True)
            if f:
                print("trying to extract",flush=True)
                f.extractall()
                print("Unpack 3 ", flush=True)

        except:
            print("Error extracting files!")

        finally:
            #os.chdir(cwd)
            print("Unpack 4 ", flush=True)

    def dispatcher(self):

        #cwd = os.getcwd()
        print("Dispatcher 1 ", flush=True)
        os.chdir(self.dir)

        try:
            cfg = None
            with open(os.path.join(self.dir, 'fedn.yaml'), 'rb') as config_file:
                import json
                cfg = json.loads(config_file.read())
                self.dispatch_config = cfg
                print("Dispatcher 2 ", flush=True)
        except Exception as e:
            print("Error trying to load and unpack dispatcher config - trying default", flush=True)
        #finally:
        #    os.chdir(cwd)
        print("Dispatcher 3 ", flush=True)

        return Dispatcher(self.dispatch_config, self.dir)
