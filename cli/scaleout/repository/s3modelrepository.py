

from scaleout.repository.miniorepository import MINIORepository


class S3ModelRepository(MINIORepository):

    def __init__(self,config):
        super().__init__(config)

    def get_model(self, model_id):
        print("Client {} trying to get model with id: {}".format(self.client, model_id),flush=True)
        return self.get_artifact(model_id)

    def set_model(self, model, bucket='alliance', is_file=True):
        import uuid
        model_id = uuid.uuid4()
        # TODO: Check that this call succeeds
        try:
            self.set_artifact(str(model_id), model,bucket=bucket,is_file=is_file)
        except Exception as e:
            print("Failed to write model with ID {} to repository.".format(model_id))
            raise
        return str(model_id)
