import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import random
from datasets import Dataset,load_dataset



# 1. 统计标签频率
def compute_label_freq(dataset):
    label_counter = Counter()
    for example in dataset:
        labels = set([r["label"] for r in example["risk_labels"]])
        label_counter.update(labels)
    return label_counter

# 2. 稀有标签筛选
def get_rare_labels(label_counter, threshold=1000):
    return {label for label, count in label_counter.items() if count < threshold}

# 3. 建立标签 -> 样本映射
def build_label_to_samples(dataset):
    label_to_examples = defaultdict(list)
    for example in dataset:
        labels = set([r["label"] for r in example["risk_labels"]])
        for label in labels:
            label_to_examples[label].append(example)
    return label_to_examples

# 4. 非线性过采样：越稀有补得越多
def oversample_rare_labels(dataset, rare_labels, label_counter, freq_threshold=1000, max_target=1000, power=1.5):
    label_to_examples = build_label_to_samples(dataset)
    new_samples = []

    for label in rare_labels:
        count = label_counter[label]
        examples = label_to_examples[label]

        # Nonlinear amplification strategy (you can adjust the power control amplitude)
        ratio = max(0.0, 1.0 - count / freq_threshold)
        target_count = int(max_target * (ratio ** power))

        needed = target_count
        if needed > 0:
            new_samples.extend(random.choices(examples, k=needed))

    print(f"The number of oversampling files that has beed generated: {len(new_samples)}")
    return Dataset.from_list(list(dataset) + new_samples)

# === 主函数入口 ===
def balance_multilabel_dataset(dataset, freq_threshold=1000, target_per_label=1000):
    label_counts = compute_label_freq(dataset)
    rare_labels = get_rare_labels(label_counts, threshold=freq_threshold)
    print(f"📉 稀有标签数: {len(rare_labels)}")
    return oversample_rare_labels(dataset, rare_labels, label_counts, freq_threshold, target_per_label)


def count_labels(dataset):
    counter = Counter()
    for item in dataset:
        risk_types = set([r["label"] for r in item["risk_labels"]])
        counter.update(risk_types)
    return counter

def comparison_datasets(train_dataset,balanced_dataset):

    orig_dist = count_labels(train_dataset)
    resampled_dist = count_labels(balanced_dataset)

    # 可视化对比
    labels = sorted(orig_dist.keys())
    x = range(len(labels))

    plt.figure(figsize=(12, 4))
    plt.bar(x, [orig_dist[l] for l in labels], label="Original")
    plt.bar(x, [resampled_dist[l] for l in labels], alpha=0.6, label="Resampled")
    plt.xticks(x, labels, rotation=45)
    plt.title("Risk Type Distribution Before and After Oversampling")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    train_dataset = load_dataset("json", data_files="./train.json")["train"]

    balanced_dataset = balance_multilabel_dataset(train_dataset)

        # 将 HuggingFace Dataset 转为 pandas DataFrame
    df = balanced_dataset.to_pandas()

    # 保存为数组形式的 JSON（列表而非 jsonl）
    df.to_json("train_balanced.json", orient="records", force_ascii=False, indent=2)

    comparison_datasets(train_dataset,balanced_dataset)

