import os
import uuid
from io import BytesIO


class LocalFileSystemModelRepository:
    def __init__(self, directory='./'):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_model_path(self, model_id):
        return os.path.join(self.directory, model_id)

    def get_model(self, model_id):
        model_path = self.get_model_path(model_id)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                return file.read()
        else:
            raise FileNotFoundError(f"Model with ID {model_id} not found.")

    def get_model_stream(self, model_id):
        model_path = self.get_model_path(model_id)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                byte_io = BytesIO(file.read())
                return byte_io
        else:
            raise FileNotFoundError(f"Model with ID {model_id} not found.")

    def set_model(self, model, is_file=True):
        model_id = str(uuid.uuid4())
        model_path = self.get_model_path(model_id)

        if is_file:
            # If model is a file path, copy the file content.
            with open(model, 'rb') as src, open(model_path, 'wb') as dst:
                dst.write(src.read())
        else:
            # If model is a binary content, write it directly.
            with open(model_path, 'wb') as file:
                file.write(model)

        return model_id

    # Methods for compute_package can be similarly implemented
    def set_compute_package(self, name, compute_package, is_file=True):
        package_path = self.get_model_path(name)
        if is_file:
            with open(compute_package, 'rb') as src, open(package_path, 'wb') as dst:
                dst.write(src.read())
        else:
            with open(package_path, 'wb') as file:
                file.write(compute_package)

    def get_compute_package(self, compute_package):
        package_path = self.get_model_path(compute_package)
        if os.path.exists(package_path):
            with open(package_path, 'rb') as file:
                return file.read()
        else:
            raise FileNotFoundError(f"Compute package {compute_package} not found.")

    def delete_compute_package(self, compute_package):
        package_path = self.get_model_path(compute_package)
        if os.path.exists(package_path):
            os.remove(package_path)
        else:
            raise FileNotFoundError(f"Compute package {compute_package} not found.")
        
    def delete_model(self, model_id):
        model_path = self.get_model_path(model_id)
        if os.path.exists(model_path):
            os.remove(model_path)
        else:
            raise FileNotFoundError(f"Model with ID {model_id} not found.")
