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


def drop_mongodb():
    config = get_mongo_config()
    try:
        mc = pymongo.MongoClient(**config)
        mdb = mc[os.environ['ALLIANCE_UID']]
        mc.drop_database(mdb)
    except Exception:
        raise
