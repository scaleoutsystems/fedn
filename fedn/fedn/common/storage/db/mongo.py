import os

import pymongo


def get_mongo_config():
    config = {
        'username': os.environ.get('FEDN_MONGO_USER', 'default'),
        'password': os.environ.get('FEDN_MONGO_PASSWORD', 'password'),
        'host': os.environ.get('FEDN_MONGO_HOST', 'localhost'),
        'port': int(os.environ.get('FEDN_MONGO_PORT', '27017')),
    }
    return config


def connect_to_mongodb(network_id):
    config = get_mongo_config()
    try:
        mc = pymongo.MongoClient(**config)
        # This is so that we check that the connection is live
        mc.server_info()
        # TODO
        mdb = mc[network_id]
        return mdb
    except Exception:
        raise
