import json


def dump_to_file(data, name, path):
    success = True
    try:
        with open(path + '/' + name + '.json', 'w') as outfile:
            json.dump(data, outfile)
    except ValueError as e:
        success = False

    if not success:
        import pickle
        try:
            with open(path + name + '.pkl', 'wb') as outfile:
                pickle.dump(data, outfile)
        except ValueError as e:
            success = False

    return success


def load_from_file(name, path):
    success = True
    data = None
    try:
        with open(path + '/' + name + '.json', 'r') as infile:
            data = json.load(infile)
    except ValueError as e:
        success = False

    if not success:
        import pickle
        try:
            with open(path + '/' + name + '.pkl', 'rb') as infile:
                data = pickle.load(infile)
        except ValueError as e:
            success = False

    return data, success
