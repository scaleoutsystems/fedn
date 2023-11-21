import uuid

from .miniorepo import MINIORepository


class S3ModelRepository(MINIORepository):
    """

    """

    def __init__(self, config):
        super().__init__(config)

    def get_model(self, model_id):
        """

        :param model_id:
        :return:
        """
        print("Client {} trying to get model with id: {}".format(
            self.client, model_id), flush=True)
        return self.get_artifact(model_id)

    def get_model_stream(self, model_id):
        """

        :param model_id:
        :return:
        """
        print("Client {} trying to get model with id: {}".format(
            self.client, model_id), flush=True)
        return self.get_artifact_stream(model_id)

    def set_model(self, model, is_file=True):
        """

        :param model:
        :param is_file:
        :return:
        """
        model_id = uuid.uuid4()
        # TODO: Check that this call succeeds
        try:
            self.set_artifact(str(model_id), model,
                              bucket=self.bucket, is_file=is_file)
        except Exception:
            print("Failed to write model with ID {} to repository.".format(model_id))
            raise
        return str(model_id)

    def set_compute_package(self, name, compute_package, is_file=True):
        """

        :param name:
        :param compute_package:
        :param is_file:
        """
        try:
            self.set_artifact(str(name), compute_package,
                              bucket="fedn-context", is_file=is_file)
        except Exception:
            print("Failed to write compute_package to repository.")
            raise

    def get_compute_package(self, compute_package):
        """

        :param compute_package:
        :return:
        """
        try:
            data = self.get_artifact(compute_package, bucket="fedn-context")
        except Exception:
            print("Failed to get compute_package from repository.")
            raise
        return data

    def delete_compute_package(self, compute_package):
        """

        :param compute_package:
        """
        try:
            self.delete_artifact(compute_package, bucket=['fedn-context'])
        except Exception:
            print("Failed to delete compute_package from repository.")
            raise
