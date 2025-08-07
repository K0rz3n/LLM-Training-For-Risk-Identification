import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_combined_label_distribution(df_train, df_val, df_test):
    # 提取所有标签
    all_labels = sorted(set(
        label for labels in pd.concat([df_train["label_list"], df_val["label_list"], df_test["label_list"]]) 
        for label in labels
    ))
    
    label2idx = {label: idx for idx, label in enumerate(all_labels)}

    # 统计各数据集中每个标签出现的次数
    def count_labels(df):
        counts = np.zeros(len(all_labels))
        for labels in df["label_list"]:
            for label in labels:
                counts[label2idx[label]] += 1
        return counts

    train_counts = count_labels(df_train)
    val_counts = count_labels(df_val)
    test_counts = count_labels(df_test)

    # 绘制并列柱状图
    x = np.arange(len(all_labels))
    width = 0.25  # 每个柱子的宽度

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, train_counts, width, label="Train set")
    ax.bar(x, val_counts, width, label="Val set")
    ax.bar(x + width, test_counts, width, label="Test set")

    ax.set_ylabel('label appearance counts')
    ax.set_title('Label distribution comparison（Train / Val / Test）')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()


# 读取数据（如果你已经有 df_train, df_val, df_test 变量了就直接用）
df_train = pd.read_json("/Users/k0rz3n/projects/individualProject/QWEN_Classfier/data/splits/second/train.json")
df_val = pd.read_json("/Users/k0rz3n/projects/individualProject/QWEN_Classfier/data/splits/second/val.json")
df_test = pd.read_json("/Users/k0rz3n/projects/individualProject/QWEN_Classfier/data/splits/second/test.json")

# 使用
plot_combined_label_distribution(df_train, df_val, df_test)