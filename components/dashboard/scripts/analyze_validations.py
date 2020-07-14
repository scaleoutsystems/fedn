import pymongo 
from bson.objectid import ObjectId
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy

c = pymongo.MongoClient()
ALLIANCE_ID="a1bdd2282-930a-44a8-96a3-15c05b5b34cb"
mc = pymongo.MongoClient('localhost',27017,username='root',password='example')
print(mc.database_names())
mdb = mc[ALLIANCE_ID]
print(mdb.collection_names())
alliance = mdb["status"]
print(alliance)

# Assemble a dict with all validations for all global model IDs
validations = {}
for post in alliance.find({'type': 'MODEL_VALIDATION'}):
	e = json.loads(post['data'])

	try: 
		validations[e['modelId']].append(json.loads(e['data'])["accuracy"])
	except KeyError:
		validations[e['modelId']] = [json.loads(e['data'])["accuracy"]]
print(validations)


x = []
y = []
for model_id, acc in validations.items():
	x.append(model_id)
	y.append(numpy.mean(acc))

rounds = range(len(y))
plt.plot(rounds,y)
plt.title("Model ID: {}".format(ALLIANCE_ID))
plt.ylabel("Mean accuracy")
plt.xlabel("Round ID")
plt.savefig('validations.png')


