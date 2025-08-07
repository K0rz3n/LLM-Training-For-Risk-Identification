import os
import json


def getfiletext(filepath):
    with open(filepath, 'r') as f:
        return f.read()


def combine_json(filedir,jsonpath):

    with open(jsonpath, 'r') as f:
        filelist = json.load(f)
    
    for file in filelist:
        filepath = os.path.join(filedir, file["filename"])
        if os.path.exists(filepath):
            filetext = getfiletext(filepath)
            file["text"] = filetext
    
    with open("./newjson.json", 'w') as f:
        json.dump(filelist, f, indent=4)
            

if __name__ == "__main__":

    filedir = "/Users/k0rz3n/sectools/docker_tools/dockerfiletest/deduplicated-sources"
    jsonpath = "/Users/k0rz3n/sectools/docker_tools/dockerfiletest/data_labeling/batch_3/batch_3.json"

    combine_json(filedir,jsonpath)
