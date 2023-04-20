import json

with open('retinanet-examples/Datasets/annotations/exhibition_setup.json', 'r') as jsonFile:
    data = json.load(jsonFile)

num = 0
for annotation in data["annotations"]:
    annotation["id"] = num
    num += 1
    original = annotation["image_id"]
    annotation["image_id"] = int(original.replace("exhibition_setup_00",""))

for image in data["images"]:
    original = image["id"]
    image["id"] = int(original.replace("exhibition_setup_00",""))

with open('retinanet-examples/Datasets/annotations/exhibition_setup.json', 'w') as jsonFile:
    json.dump(data, jsonFile)