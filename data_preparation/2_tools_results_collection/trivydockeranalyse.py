import os 
import subprocess
import json
import shutil
import threading
import random

include_id = ["DS001", "DS002", "DS004", "DS005", "DS006","DS007", "DS008", "DS009", "DS010", "DS011","DS012", "DS013", "DS016", "DS017", "DS021","DS024", "DS029", "DS030", "DS031","DS014","DS022"]

def check_dir(result_dir):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    print("Directory trivy_result created.")

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
    
def run_trivy(filepath):
    try:
        ret = subprocess.run(
            ["trivy", "config", "--misconfig-scanners=dockerfile", "--format", "json", filepath],
            text=True,
            capture_output=True
            )
        result_dict = json.loads(ret.stdout)['Results'][0]["Misconfigurations"]
        new_lst = []
        for misconfiguration in result_dict:
            if misconfiguration['ID'] in include_id:
                new_lst.append(misconfiguration)
        result_dict = new_lst

        # print(result_dict)
        
        return result_dict
    

        # return ret.stdout  # Returning the raw output of trivy for further processing or analysis.
    except Exception as e:
        print(f"scan got error {e}, filename:{filepath}")
        return None


def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def scan_dockerfiles(dockerfiles, output_file):
    results = {}
    for dockerfile in dockerfiles:
        filename = os.path.split(dockerfile)[1]
        risk  = run_trivy(dockerfile)
        if risk:
            results[filename] = risk
        else:
            pass
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Trivy scan completed. Results saved to {output_file}")

def combine_results(result_dir,final_result):
    result_files = os.listdir(result_dir)
    results = {}
    for result_file in result_files:
        with open(os.path.join(result_dir, result_file), "r") as f:
            results.update(json.load(f))
    with open(os.path.join(result_dir, final_result), "w") as f:
        json.dump(results, f, indent=4)
    print(f"All trivy scan results combined. Results saved to {result_dir}/{final_result}))")

def muti_process(directory,number,selected_files,chunk_size,result_dir,result_file):
    dockerfiles = finddockerfiles(directory,number,selected_files)
    dockerfile_silces = chunk_list(dockerfiles, chunk_size)
    threads = []  # 用于存储所有线程
    for k,v in enumerate(dockerfile_silces):
        t = threading.Thread(target=scan_dockerfiles, args=(v,os.path.join(result_dir,f"trivy_results_{k}.json")))
        t.start()
        threads.append(t)
        print(f"Thread{k} created successfully")
    
    for t in threads:
        t.join()  # 主线程需要等所有子线程执行结束才结束
        print(f"Thread{t.name} completed successfully.")  # 打印线程执行结束的消息

    print("All threads completed succeed.")
    combine_results(result_dir,result_file)


if __name__ == '__main__':
    # result_dir = "./trivy_result"
    # final_result = "trivy_results.json"
    # directory = "./test1"
    # directory = "./deduplicated-sources"
    # selected_files = "./selected_dockerfiles.json"
    # selected_number = 40000

    result_dir = "./trivy_without_risk_test"
    final_result = "trivy_results.json"
    directory = "./data_scrubbing/target_files_2"
    selected_files = "./selected_without_risk_dockerfiles_2.json"
    selected_number = 100
    chunk = 50
    check_dir(result_dir)
    muti_process(directory,selected_number,selected_files,chunk,result_dir,final_result)

  
