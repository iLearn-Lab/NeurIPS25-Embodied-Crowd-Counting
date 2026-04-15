import json   

def load_flight_data(path):
    
    with open(path, 'r') as f:
        json_data = json.load(f)
    flight_data = dict()
    for key in json_data.keys():
        flight_data[key] = []
        for item in json_data[key]:
            flight_data[key].append(tuple(item))

    return flight_data