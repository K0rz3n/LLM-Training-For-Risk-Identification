import os
import pymysql

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'individual_project'
}

# 读取目录中的文件并插入到数据库
def store_filenames_to_mysql(directory_path):
    # 创建数据库连接
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 确保目标表存在
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL
        )
    """)

    # 读取目录中的所有文件
    for filename in os.listdir(directory_path):
        # 仅处理文件，不处理子目录
        if os.path.isfile(os.path.join(directory_path, filename)):
            # 插入文件名到数据库
            cursor.execute("""
                INSERT INTO files (filename)
                VALUES (%s)
            """, (filename,))

    # 提交事务并关闭连接
    conn.commit()
    cursor.close()
    conn.close()

    print("Filenames inserted successfully!")


if __name__ == "__main__":

    # 目录路径
    directory_path = "/Users/k0rz3n/sectools/docker_tools/dockerfiletest/testdata"

    # 调用函数
    store_filenames_to_mysql(directory_path)