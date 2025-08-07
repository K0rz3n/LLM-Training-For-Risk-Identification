import os 
import subprocess
import json
import shutil
import threading
import random
from concurrent.futures import ThreadPoolExecutor

# include_id = ["0aedd324", "2e92d18c", "2064113b", "a248d89e", "bab38efd","4f469f06", "22f535ec", "96f59ca3", "03be1867", "79731185","f445bd25", "edd9f7d3", "c4f2e24a", "c923ad4b", "fed3d812","bfe0be8b", "ba0a34dc", "d859b2eb", "19d4cfc7","e0e1edad","37db3a53","8bd60033","22261deb","9d9cbf83"]


include_id = ["0aedd324","2e92d18c","4f469f06","f445bd25","edd9f7d3","c4f2e24a","19d4cfc7","e0e1edad","37db3a53","22261deb"]



def check_dir(result_dir,temp_dir):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(result_dir)
    print("Directory kics_result created.")
    os.makedirs(temp_dir)
    print("Directory kics_temp created.")




def finddockerfiles(directory,number,selected_files):
    dockerfiles = []
    if os.path.exists(selected_files):
        print(f"{selected_files} existing")
        with open(selected_files, "r") as f:
            dockerfiles = json.load(f)
    else:
        for root, _, files in os.walk(directory):
            for file in files:
                dockerfiles.append(os.path.join(root, file))
        dockerfiles = random.sample(dockerfiles,number)
        with open(selected_files, "w") as f:
            json.dump(dockerfiles, f)
        print(f"{selected_files} is not existing, create succeed")

    return dockerfiles


def run_kicsscan(filepath,temp_dir,result_name):
    try:
        subprocess.run(
            ["kics", "scan", "--report-formats", "json", "--config", "../kics_config/kicsconfig.yaml", "--path", filepath, "--output-path", temp_dir, "--output-name", result_name],
            text=True,
            capture_output=True
            )
        with open(os.path.join(temp_dir,result_name),"r") as f:
            res = json.loads(f.read())
        result = []
        if res:
            for q in res["queries"]:
                result.append(
                    {
                        "query_name": q["query_name"],
                        "description": q["description"],
                        "description_id": q["description_id"],
                        "severity": q["severity"],
                        "query_id": q["query_id"],
                        "query_url": q["query_url"],
                        "file": os.path.split(q["files"][0]["file_name"])[1],
                        "line": ",".join(sorted(list(set([str(file["line"]) for file in q["files"]])))),
                    }
                )
            new_lst = []
            for misconfiguration in result:
                if misconfiguration['description_id'] in include_id:
                    new_lst.append(misconfiguration)
            return new_lst
        else:
            return result

    
    except Exception as e:
        print(f"scan got error {e}, filename:{filepath}")
        return None


def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def scan_dockerfiles(dockerfiles, output_file, temp_dir,sequence):
    print(f"Thread{sequence} created successfully")
    results = {}
    for dockerfile in dockerfiles:
        filename = os.path.split(dockerfile)[1]
        risk  = run_kicsscan(dockerfile,temp_dir,f"result_{sequence}.json")
        if risk:
            results[filename] = risk
        else:
            pass
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    results.clear()
    print(f"Thread {sequence} completed successfully. Results saved to {output_file}")

def combine_results(result_dir,final_result):
    result_files = os.listdir(result_dir)
    results = {}
    for result_file in result_files:
        with open(os.path.join(result_dir, result_file), "r") as f:
            results.update(json.load(f))
    with open(os.path.join(result_dir, final_result), "w") as f:
        json.dump(results, f, indent=4)
    print(f"All Kics scan results combined. Results saved to {result_dir}/{final_result}))")



def muti_process(directory,number,selected_files,chunk_size,result_dir,result_file,temp_dir):
    dockerfiles = finddockerfiles(directory,number,selected_files)
    dockerfile_slices = chunk_list(dockerfiles, chunk_size)
    # threads = []  # 用于存储所有线程
    MAX_THREADS = 10
    # for k,v in enumerate(dockerfile_silces):
    #     t = threading.Thread(target=scan_dockerfiles, args=(v,os.path.join(result_dir,f"kics_results_{k}.json"),temp_dir,k))
    #     t.start()
    #     threads.append(t)
    #     print(f"Thread{k} created successfully")
    
    # for t in threads:
    #     t.join()  # 主线程需要等所有子线程执行结束才结束
    #     print(f"Thread{t.name} completed successfully.")  # 打印线程执行结束的消息

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        executor.map(scan_dockerfiles, 
                 dockerfile_slices, 
                 [os.path.join(result_dir, f"kics_results_{k}.json") for k in range(len(dockerfile_slices))],
                 [temp_dir] * len(dockerfile_slices), 
                 range(len(dockerfile_slices)))

    print("All threads completed succeed.")
    combine_results(result_dir,result_file)


    

if __name__ == "__main__":
    # directory = "./testdata"
    # directory = "./deduplicated-sources"
    # result_dir = "./kics_result"
    # final_result = "kics_results.json"
    # temp_dir = "./kics_temp"
    # selected_files = "selected_dockerfiles.json"
    # selected_number = 40000
    # chunk = 100

    # result_dir = "./kics_without_risk_test"
    # final_result = "kics_results.json"
    # directory = "./data_scrubbing/target_files_2"
    # selected_files = "./selected_without_risk_dockerfiles_2.json"
    # temp_dir = "./kics_temp_2"
    # selected_number = 100
    # chunk = 50

    result_dir = "./kics_without_risk_test"
    final_result = "kics_results.json"
    directory = "/Users/k0rz3n/projects/individualProject/QWEN_Classfier/data_eval/eval_files"
    selected_files = "./selected_eval_dockerfiles.json"
    temp_dir = "./kics_temp_2"
    selected_number = 215
    chunk = 50

    check_dir(result_dir,temp_dir)
    muti_process(directory,selected_number,selected_files,chunk,result_dir,final_result,temp_dir)