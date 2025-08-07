import json
import os
import random

def read_data(filepath):
    with open(filepath,"r") as f:
        data = json.loads(f.read())
    # print(data)
    return data


def read_files(directory):

    data_frame = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    data_frame.append({
                    "filename":file,
                    "file_content":content,
                    "risk_labels":[
                        {
                        "label":"dockerfile-without-risk",
                        "start":-1,
                        "end":-1,
                        "text":"The dockerfile has been scanned and no security risks were detected."
                        }
                    ]
                })
            except Exception as e:
                print(f"Fail to process data {file_path}: {e}")
    return data_frame


def genDF_for_standard(result_path_1,result_path_2):
    
    data_frame = []
    
    data_1 = read_data(result_path_1)
    data_2 = read_data(result_path_2)

    data_lists = [data_1,data_2]

    for data in data_lists:
        for file in data:
            filename  = file["filename"]
            label_id = file["id"]
            file_content = file["text"]
            label_list = []
            if "label" in file:
                for label in file["label"]:
                    label_name = label["labels"][0]
                    if label["labels"][0] == "root-privilege-user" and not label["text"].lower().startswith("user"):
                        start = -1
                        end = -1
                        risk_text = "The USER directive was not explicitly declared; the process will run as the default root user."
                    else:
                        start = label["start"]
                        end = label["end"]
                        risk_text = label["text"]
                    label_list.append({
                        "label":label_name,
                        "start":start,
                        "end":end,
                        "text":risk_text
                    })
                data_frame.append({
                    "filename":filename,
                    "file_content":file_content,
                    "risk_labels":label_list
                })
            else:
                print(filename)
                print(label_id)
                break

    with open("./data_frame.json","w") as file:
        json.dump(data_frame, file, indent=2, ensure_ascii=False)


def genDF_for_mix(result_path_1,result_path_2,filepath):
    
    data_frame = []
    
    data_1 = read_data(result_path_1)
    data_2 = read_data(result_path_2)

    data_lists = [data_1,data_2]

    for data in data_lists:
        for file in data:
            filename  = file["filename"]
            label_id = file["id"]
            file_content = file["text"]
            label_list = []
            if "label" in file:
                for label in file["label"]:
                    label_name = label["labels"][0]
                    if label["labels"][0] == "root-privilege-user" and not label["text"].lower().startswith("user"):
                        start = -1
                        end = -1
                        risk_text = "The USER directive was not explicitly declared; the process will run as the default root user."
                    else:
                        start = label["start"]
                        end = label["end"]
                        risk_text = label["text"]
                    label_list.append({
                        "label":label_name,
                        "start":start,
                        "end":end,
                        "text":risk_text
                    })
                data_frame.append({
                    "filename":filename,
                    "file_content":file_content,
                    "risk_labels":label_list
                })
            else:
                print(filename)
                print(label_id)
                break
    
    # subset = random.sample(data_frame,749)
    file_dataframe = read_files(filepath)

    data_frame.extend(file_dataframe)

    # print(len(subset))


    with open("./data_frame_for_file.json","w") as file:
        json.dump(data_frame, file, indent=2, ensure_ascii=False)






if __name__ == "__main__":
    result_path_1 = "/Users/k0rz3n/projects/individualProject/docker_tools/dockerfiletest/data_labeling/batch_1/result_1.json"
    result_path_2 = "/Users/k0rz3n/projects/individualProject/docker_tools/dockerfiletest/data_labeling/batch_3/result_3.json"
    # genDF_for_standard(result_path_1,result_path_2)


    filepath = "/Users/k0rz3n/projects/individualProject/docker_tools/dockerfiletest/data_scrubbing/combine_together"

    genDF_for_mix(result_path_1,result_path_2,filepath)
    

    