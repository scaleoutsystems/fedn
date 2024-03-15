import tarfile
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
client_path = os.path.join(dir_path, '..', 'client')
sys.path.append(os.path.abspath(client_path))

import model

model.init_seed()

# Define the name of the archive and the directory to archive
archive_name = "package.tgz"
directory_to_archive = "client"

# Create a .tgz archive
with tarfile.open(archive_name, "w:gz") as tar:
    tar.add(directory_to_archive,
            arcname=os.path.basename(directory_to_archive))
