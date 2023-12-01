import pymongo


def connect_to_mongodb(config, network_id):
    """ Establish client connection to MongoDB.

    :param config: Dictionary containing connection strings and security credentials.
    :type config: dict
    :param network_id: Unique identifier for the FEDn network, used as db name
    :type network_id: str
    :return: MongoDB client pointing to the db corresponding to network_id
    """
    try:
        mc = pymongo.MongoClient(**config)
        # This is so that we check that the connection is live
        mc.server_info()
        mdb = mc[network_id]
        return mdb
    except Exception:
        raise
