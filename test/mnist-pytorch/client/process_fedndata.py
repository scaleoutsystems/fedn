import json
import numpy as np
def get_training_results(path, metric = 'test_accuracy'):
    with open(path) as f:
      json_data = json.load(f)
    model_validation = [x for x in json_data if x['type'] == 'MODEL_VALIDATION']
    validations = {}
    for post in model_validation:
        e = json.loads(post['data'])
        try:
            validations[e['modelId']].append(float(json.loads(e['data'])[metric]))
        except KeyError:
            validations[e['modelId']] = [float(json.loads(e['data'])[metric])]
    model_ids = []
    metric_results = []
    for model_id, acc in validations.items():
        model_ids.append(model_id)
        metric_results.append(np.mean(acc))
    return model_ids, metric_results