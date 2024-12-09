from abc import ABC, abstractmethod


class ModelStorage(ABC):
    @abstractmethod
    def exist(self, model_id):
        """Check if model exists in storage

        :param model_id: The model id
        :type model_id: str
        :return: True if model exists, False otherwise
        :rtype: bool
        """
        pass

    @abstractmethod
    def get(self, model_id):
        """Get model from storage

        :param model_id: The model id
        :type model_id: str
        :return: The model
        :rtype: object
        """
        pass

    @abstractmethod
    def get_model_metadata(self, model_id):
        """Get model metadata from storage

        :param model_id: The model id
        :type model_id: str
        :return: The model metadata
        :rtype: dict
        """
        pass

    @abstractmethod
    def set_model_metadata(self, model_id, model_metadata):
        """Set model metadata in storage

        :param model_id: The model id
        :type model_id: str
        :param model_metadata: The model metadata
        :type model_metadata: dict
        :return: True if successful, False otherwise
        :rtype: bool
        """
        pass

    @abstractmethod
    def delete(self, model_id):
        """Delete model from storage

        :param model_id: The model id
        :type model_id: str
        :return: True if successful, False otherwise
        :rtype: bool
        """
        pass

    @abstractmethod
    def delete_all(self):
        """Delete all models from storage

        :return: True if successful, False otherwise
        :rtype: bool
        """
        pass
