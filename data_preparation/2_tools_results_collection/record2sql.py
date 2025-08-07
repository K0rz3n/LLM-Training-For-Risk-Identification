import pymysql
import json

# 数据库连接信息
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "individual_project",
    "port": 3306
}


def read_data(filepath):
    with open(filepath,"r") as f:
        data = json.loads(f.read())
    # print(data)
    return data

def store_hadrisks_to_mysql(data):
    # 创建数据库连接
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 创建表格
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hadrisks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            risk_id VARCHAR(255) NOT NULL,
            risk_level VARCHAR(255) NOT NULL,
            risk_line VARCHAR(255) NOT NULL,
            description TEXT NOT NULL
            
        )
    """)

    # 插入数据
    for filename, risks in data.items():
        for risk in risks:
            cursor.execute("""
                INSERT INTO hadrisks (filename, risk_id, risk_level, risk_line, description)
                VALUES (%s, %s, %s, %s, %s)
            """, (filename, risk["code"], risk["level"], risk["line"], risk["message"]))

    print("Insert hadolint successfully")

    # 关闭数据库连接
    conn.commit()
    cursor.close()
    conn.close()

def store_terrrisks_to_mysql(data):
    # 创建数据库连接
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 创建表格
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS terrrisks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            risk_id VARCHAR(255) NOT NULL,
            risk_name VARCHAR(255) NOT NULL,
            risk_level VARCHAR(255) NOT NULL,
            risk_line VARCHAR(255) NOT NULL,
            description TEXT NOT NULL
        )
    """)

    # 插入数据
    for filename, risks in data.items():
        for risk in risks:
            cursor.execute("""
                INSERT INTO terrrisks (filename, risk_id, risk_name, risk_level, risk_line, description)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (filename, risk["rule_id"], risk["rule_name"], risk["severity"], risk["line"], risk["description"]))

    # 关闭数据库连接
    conn.commit()
    cursor.close()
    conn.close()




def store_trivyrisks_to_mysql(data):
    # 创建数据库连接
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 创建表格
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trivyrisks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            risk_id VARCHAR(255) NOT NULL,
            risk_name VARCHAR(255) NOT NULL,
            risk_level VARCHAR(255) NOT NULL,
            risk_line VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            resolution TEXT NOT NULL
        )
    """)

    # 插入数据
    for filename, risks in data.items():
        for risk in risks:
            # 使用 .get() 来避免 KeyError 错误，如果找不到则返回默认值
            start_line = risk.get("CauseMetadata", {}).get("StartLine", None)
            end_line = risk.get("CauseMetadata", {}).get("EndLine", None)
            if start_line == end_line and start_line and end_line:
                risk_line = start_line
            elif start_line != end_line and start_line and end_line:
                risk_line = f"{start_line}-{end_line}"
            else:
                risk_line = "N/A"
            
            cursor.execute("""
                INSERT INTO trivyrisks (filename, risk_id, risk_name, risk_level, risk_line, description, resolution)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (filename, risk["ID"], risk["Title"], risk["Severity"],risk_line, risk["Description"], risk["Resolution"]))

    print("Insert trivy successfully")
    
    conn.commit()
    cursor.close()
    conn.close()



def store_kicsrisks_to_mysql(data):
    # 创建数据库连接
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 创建表格
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kicsrisks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            risk_id VARCHAR(255) NOT NULL,
            query_id VARCHAR(255) NOT NULL,
            risk_name VARCHAR(255) NOT NULL,
            risk_level VARCHAR(255) NOT NULL,
            risk_line TEXT NOT NULL,
            description TEXT NOT NULL,
            query_url VARCHAR(255) NOT NULL
        )
    """)

    # 插入数据
    for filename, risks in data.items():
        for risk in risks:
   
            cursor.execute("""
                INSERT INTO kicsrisks (filename, risk_id, query_id, risk_name, risk_level, risk_line, description, query_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (filename, risk["description_id"], risk["query_id"], risk["query_name"],risk["severity"], risk["line"], risk["description"],risk["query_url"]))

    print("Insert kics successfully")
    
    conn.commit()
    cursor.close()
    conn.close()



if __name__ == "__main__":
    # had_filepath = "./hadolint_result/hadolint_results.json"
    # had_data = read_data(had_filepath)
    # store_hadrisks_to_mysql(had_data)

    # terr_filepath = "./terrascan_results.json"
    # terr_data = read_data(terr_filepath)
    # store_terrrisks_to_mysql(terr_data)

    trivy_filepath = "./trivy_result/trivy_results.json"
    trivy_data = read_data(trivy_filepath)
    store_trivyrisks_to_mysql(trivy_data)

    # kics_filepath = "./kics_result/kics_results.json"
    # kics_data = read_data(kics_filepath)
    # store_kicsrisks_to_mysql(kics_data)

