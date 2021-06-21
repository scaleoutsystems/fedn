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
            # for root, dirs, files in os.walk(self.file_path):
            # for file in files:
            # tf.add(os.path.join(root, file))
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
            # data = {'name': self.package_file, 'hash': str(self.package_hash)}
            # print("going to send {}".format(data),flush=True)
            f = open(os.path.join(os.path.dirname(self.file_path), self.package_file), 'rb')
            print("Sending the following file {}".format(f.read()), flush=True)
            f.seek(0, 0)
            files = {'file': f}
            try:
                retval = requests.post('https://{}:{}/context'.format(self.reducer_host, self.reducer_port),
                                       verify=False, files=files,
                                       # data=data,
                                       headers={'Authorization': 'Token {}'.format(self.reducer_token)})
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
        self.checksum = None
        self.expected_checksum = None

    def download(self, host, port, token, name=None):
        import requests

        path = "https://{}:{}/context".format(host, port)
        if name:
            path = path + "?name={}".format(name)

        with requests.get(path, stream=True, verify=False, headers={'Authorization': 'Token {}'.format(token)}) as r:
            if 200 <= r.status_code < 204:
                import cgi
                params = cgi.parse_header(r.headers.get('Content-Disposition', ''))[-1]
                try:
                    self.pkg_name = params['filename']
                except KeyError:
                    print("No package returned!", flush=True)
                    return None
                r.raise_for_status()
                with open(os.path.join(self.pkg_path, self.pkg_name), 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)


        path = "https://{}:{}/checksum".format(host, port)

        if name:
            path = path + "?name={}".format(name)
        with requests.get(path, verify=False, headers={'Authorization': 'Token {}'.format(token)}) as r:
            if 200 <= r.status_code < 204:

                data = r.json()
                try:
                    self.checksum = data['checksum']
                except Exception as e:
                    print("Could not extract checksum!")


        return True

    def validate(self, expected_checksum):

        self.expected_checksum = expected_checksum

        from fedn.utils.checksum import md5

        # crosscheck checksum and unpack if security checks are ok.
        #print("check if checksum {} is equal to checksum expected {}".format(self.checksum,self.expected_checksum))
        file_checksum = str(md5(os.path.join(self.pkg_path,self.pkg_name)))

        # catched by client, make configurable by governance network!
        #if self.expected_checksum is None:
        #    print("CAUTION: Package validation turned OFF on client", flush=True)
        #    return True

        if self.checksum == self.expected_checksum == file_checksum:
            print("Package validated {}".format(self.checksum))
            return True
        else:
            return False



    def unpack(self):
        import os
        import tarfile

        if self.pkg_name:
            f = None
            if self.pkg_name.endswith('tar.gz'):
                f = tarfile.open(os.path.join(self.pkg_path, self.pkg_name), 'r:gz')
            if self.pkg_name.endswith('.tgz'):
                f = tarfile.open(os.path.join(self.pkg_path, self.pkg_name), 'r:gz')
            if self.pkg_name.endswith('tar.bz2'):
                f = tarfile.open(os.path.join(self.pkg_path, self.pkg_name), 'r:bz2')
        else:
            print("Failed to unpack compute package, no pkg_name set. Has the reducer been configured with a compute package?")

        import os
        cwd = os.getcwd()
        try:
            os.chdir(self.dir)

            if f:
                f.extractall()
                print("Successfully extracted compute package content in {}".format(self.dir), flush=True)
        except:
            print("Error extracting files!")

    def dispatcher(self, run_path):

        dispatch_dir = self.dir
        from_path = os.path.join(os.getcwd(), 'client')
        import time
        dirname = time.strftime("%Y%m%d-%H%M%S")

        from distutils.dir_util import copy_tree
        copy_tree(from_path, run_path)

        os.chdir(run_path)


        try:
            cfg = None
            with open(os.path.join(to_path, 'fedn.yaml'), 'rb') as config_file:
                import yaml
                cfg = yaml.safe_load(config_file.read())
                self.dispatch_config = cfg

        except Exception as e:
            print("Error trying to load and unpack dispatcher config - trying default", flush=True)

        dispatcher = Dispatcher(self.dispatch_config, run_path)

        return dispatcher
