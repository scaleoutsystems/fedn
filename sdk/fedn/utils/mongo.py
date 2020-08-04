import pymongo
import os


def get_mongo_config():
	config = {
<<<<<<< HEAD
		'username':os.environ['MDBUSR'],
		'password':os.environ['MDBPWD'],
		'host':'mongo',
		'port':27017,
=======
		'username': os.environ['MDBUSR']
		'password': os.environ['MDBPWD']
		'host': 'mongo'
		'port': 27017
>>>>>>> cf289a777f47f108981b4761b0bf2f7662244162
	}
	return config 


def connect_to_mongodb():
<<<<<<< HEAD
    config = get_mongo_config()
=======
	config = get_mongo_config()
>>>>>>> cf289a777f47f108981b4761b0bf2f7662244162
    try: 
        mc = pymongo.MongoClient(**config)
        # This is so that we check that the connection is live
        mc.server_info()
        # TODO
        mdb = mc[os.environ['ALLIANCE_UID']]
        return mdb
    except Exception:
    	raise
  