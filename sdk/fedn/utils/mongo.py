import pymongo
import os

def connect_to_mongodb():
    # TODO: Get from configuration
    try: 
        # TODO: Get configs from env
        mc = pymongo.MongoClient('mongo',27017,username='root',password='example')
        # This is so that we check that the connection is live
        mc.server_info()
        # TODO
        mdb = mc[os.environ['ALLIANCE_UID']]
        return mdb
    except Exception:
    	raise
        #return None
  