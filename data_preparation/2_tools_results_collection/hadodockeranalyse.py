import os 
import subprocess
import json
import shutil
import threading
import random

include_id = ["DL3020", "DL3015", "DL3014", "DL3023", "DL3021","DL3006", "DL3007", "DL4003", "DL4004", "DL3002","DL3005", "DL3003", "DL3004", "DL3024", "DL3011","DL3000", "DL4005", "DL3033", "DL3008","DL3022","DL3030","DL3013","DL3016","DL4001","DL4000"]


def check_dir(result_dir):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    print("Directory hadolint_result created.")


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


def run_hadolint(filepath):
    try:
        ret = subprocess.run(
            ["hadolint", "--format", "json", filepath],
            text=True,
            capture_output=True
            )
        result = json.loads(ret.stdout)
        for risk in result:
            risk["file"] = os.path.split(risk["file"])[1]
        new_lst = []
        for misconfiguration in result:
            if misconfiguration['code'] in include_id:
                new_lst.append(misconfiguration)
        return new_lst
    except Exception as e:
        print(f"scan got error, filename:{filepath}")
        return None

def chunk_list(lst, chunk_size=2000):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def scan_dockerfiles(dockerfile_silce, output_file):
    
    results = {}
    for dockerfile in dockerfile_silce:
        filename = os.path.split(dockerfile)[1]
        risk  = run_hadolint(dockerfile)
        if risk:
            results[filename] = risk
        else:
            pass
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Hadolint scan completed. Results saved to {output_file}")

def combine_results(result_dir,final_result):
    result_files = os.listdir(result_dir)
    results = {}
    for result_file in result_files:
        with open(os.path.join(result_dir, result_file), "r") as f:
            results.update(json.load(f))
    with open(os.path.join(result_dir, final_result), "w") as f:
        json.dump(results, f, indent=4)
    print(f"All Hadolint scan results combined. Results saved to {result_dir}/{final_result}")


def muti_process(directory,number,selected_files,chunk_size,result_dir,final_result):
    dockerfiles = finddockerfiles(directory,number,selected_files)
    dockerfile_silces = chunk_list(dockerfiles, chunk_size)
    threads = []  # 用于存储所有线程
    for k,v in enumerate(dockerfile_silces):
        t = threading.Thread(target=scan_dockerfiles, args=(v,f"{result_dir}/hadolint_results_{k}.json"))
        t.start()
        threads.append(t)
        print(f"Thread{k} created successfully")
    
    for t in threads:
        t.join()  # 主线程需要等所有子线程执行结束才结束
        print(f"Thread{t.name} completed successfully.")  # 打印线程执行结束的消息

    print("All threads completed succeed.")
    combine_results(result_dir,final_result)


if __name__ == '__main__':
    
    # directory = "./testdata"
    # directory = "./deduplicated-sources"
    # result_dir = "./hadolint_result"
    # final_result = "hadolint_results.json"
    # selected_files = "./selected_dockerfiles.json"
    # selected_number = 40000
    # chunk = 1000

    result_dir = "./hadolint_without_risk_test"
    final_result = "hadolint_results.json"
    directory = "./data_scrubbing/target_files_2"
    selected_files = "./selected_without_risk_dockerfiles_2.json"
    selected_number = 100
    chunk = 50
    check_dir(result_dir)    
    muti_process(directory,selected_number,selected_files,chunk,result_dir,final_result)
  
