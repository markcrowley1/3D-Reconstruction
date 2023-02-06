"""
Description:
    Script used to create json file with labels mapped to synset ids
"""

import os
import json

SHAPENETCORE = "D:/shapenet_base/shapenet_core"
RENDERINGS = "D:/ShapeNetRendering/ShapeNetRendering"

def map_labels(data: list) -> dict:
    """Determine names of base categories"""
    labels = {}
    ids = {}
    instances = {}
    for name in os.listdir(SHAPENETCORE):
        path = os.path.join(SHAPENETCORE, name)
        if os.path.isdir(path):
            id = os.path.split(path)[1]
            for item in data:
                if item["synsetId"] == id:
                    ids[item["name"]] = str(item["synsetId"])
                    instances[item["name"]] = str(item["numInstances"])
                    break
    labels["numInstances"] = instances
    labels["synsetIds"] = ids
    return labels

def main():
    # Read taxonomy data
    taxonomy = os.path.join(SHAPENETCORE, "taxonomy.json")
    file = open(taxonomy, "r")
    taxonomy = json.load(file)
    file.close()

    # Create JSON file mapping synset id to label
    labels: dict = map_labels(taxonomy)
    file = open("labels.json", "w")
    json.dump(labels, file, indent=4)
    file.close()


if __name__ == "__main__":
    main()