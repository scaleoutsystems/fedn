from scaleout.repository.miniorepository import MINIORepository
from scaleout.repository.s3modelrepository import S3ModelRepository


def get_repository(config=None):
    return S3ModelRepository(config)
    #return MINIORepository(config)
