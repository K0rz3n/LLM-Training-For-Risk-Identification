import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit




# 1. 读取你的原始 JSON 数据
with open("./data_frame_for_file.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. 处理你的风险标签（提取标签列表）
# 假设每条记录 'risk_labels' 是一个 [{'label': 'xxx'}, ...] 列表
df["label_list"] = df["risk_labels"].apply(lambda risks: list({r["label"] for r in risks}))

# 3. 将标签转换成 multi-hot 矩阵（适合分层划分）
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["label_list"])

# 4. 用 Multilabel Stratified Shuffle Split 划分 80% train/val + 20% test
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
train_val_idx, test_idx = next(msss.split(df["file_content"], Y))

df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

# 5. 再从 train_val 划 5% 验证集
Y_train_val = Y[train_val_idx]
msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.0526, random_state=42)  # 5/95
train_idx, val_idx = next(msss_val.split(df_train_val["file_content"], Y_train_val))

df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val = df_train_val.iloc[val_idx].reset_index(drop=True)


# 6. 保存
df_train.to_json("train.json", orient="records", indent=2, force_ascii=False)
df_val.to_json("val.json", orient="records", indent=2, force_ascii=False)
df_test.to_json("test.json", orient="records", indent=2, force_ascii=False)

# 7. 打印确认
print(f"训练集数量: {len(df_train)}")
print(f"验证集数量: {len(df_val)}")
print(f"测试集数量: {len(df_test)}")