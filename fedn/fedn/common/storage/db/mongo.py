import os

import pymongo


def connect_to_mongodb(config, network_id):
    try:
        mc = pymongo.MongoClient(**config)
        # This is so that we check that the connection is live
        mc.server_info()
        # TODO
        mdb = mc[network_id]
        return mdb
    except Exception:
        raise


def drop_mongodb(config, network_id):
    try:
        mc = pymongo.MongoClient(**config)
        mdb = mc[network_id]
        mc.drop_database(mdb)
    except Exception:
        raise
