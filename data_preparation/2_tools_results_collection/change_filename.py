import os
import concurrent.futures

# 用来重命名文件的函数
def rename_file(file_path):
    dir_name, file_name = os.path.split(file_path)
    if file_name.endswith(".Dockerfile"):
        new_name = file_name.lower()
        new_path = os.path.join(dir_name, new_name)
        os.rename(file_path, new_path)
        print(f"Renamed: {file_path} -> {new_path}")

# 批量重命名文件
def batch_rename_files(directory):
    # 获取目录下的所有文件
    files_to_rename = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".Dockerfile"):  # 只选取 .Dockerfile 后缀的文件
                files_to_rename.append(os.path.join(root, file))
    
    # 创建线程池并执行任务
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 使用 map 方法并行处理文件重命名
        executor.map(rename_file, files_to_rename)

# 使用时指定文件夹路径
directory_path = "./deduplicated-sources"
batch_rename_files(directory_path)
