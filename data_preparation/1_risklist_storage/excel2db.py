import pandas as pd
from sqlalchemy import create_engine


file_path = "risk_list.xlsx"
df = pd.read_excel(file_path)
df = df.fillna("N/A")

engine = create_engine("mysql+pymysql://root:123456@localhost:3306/individual_project")


df.to_sql("risk_mapping", con=engine, if_exists="replace", index=False)

print("succeed!")