import pymongo
import os


def get_mongo_config():
	config = {
		'username':os.environ['MDBUSR'],
		'password':os.environ['MDBPWD'],
		'host':'mongo',
		'port':27017,
	}
	return config 


def connect_to_mongodb():
    config = get_mongo_config()
    try: 
        mc = pymongo.MongoClient(**config)
        # This is so that we check that the connection is live
        mc.server_info()
        # TODO
        mdb = mc[os.environ['ALLIANCE_UID']]
        return mdb
    except Exception:
    	raise
  