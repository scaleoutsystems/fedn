
import logging
import os

class Logger:
    def __init__(self, log_level=logging.DEBUG, to_file='',file_path=os.getcwd()):
        import sys

        root = logging.getLogger()
        root.setLevel(log_level)

        #sh = logging.StreamHandler(sys.stdout)
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        sh.setFormatter(formatter)
        root.addHandler(sh)

        if to_file != '':
            fh = logging.FileHandler(os.path.join(file_path,'{}'.format(to_file)))
            fh.setFormatter(logging.Formatter(log_format))
            root.addHandler(fh)
