import re
import json
from sklearn.metrics import precision_score, recall_score, f1_score


def parse_model_output(output_text):
    results = []
    
    matches = re.findall(
        r"Risk type:\s*(.*?)\s*Snippet:\s*['\"](.*?)['\"]\s*Position:\s*(\d+|N/A)-(\d+|N/A)", 
        output_text, 
        flags=re.DOTALL
    )

    for match in matches:
        risk_type, snippet, start, end = match

        # N/A 特判处理
        start = -1 if start == "N/A" else int(start)
        end = -1 if end == "N/A" else int(end)

        results.append({
            "risk_type": risk_type.strip(),
            "snippet": snippet.strip(),
            "start": start,
            "end": end
        })

    return results



def evaluate(ground_truths, predictions):
    """
    多标签分类评估:基于risk_type
    """
    all_labels = sorted(list({
        label["risk_type"]
        for item in ground_truths
        for label in item
    } | {
        label["risk_type"]
        for item in predictions
        for label in item
    }))

    label2id = {label: idx for idx, label in enumerate(all_labels)}

    def binarize(labels):
        vec = [0] * len(all_labels)
        for item in labels:
            if item["risk_type"] in label2id:
                vec[label2id[item["risk_type"]]] = 1
        return vec

    y_true = [binarize(gt) for gt in ground_truths]
    y_pred = [binarize(pred) for pred in predictions]

    precision = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")

    return precision, recall, f1


def load_file():

    # 加载测试集
    test_data = []
    with open("../data/evaluation_data/qwen_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))

    # 加载推理结果
    predictions_data = []
    with open("./predictions/test_predictions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            predictions_data.append(json.loads(line))

    assert len(test_data) == len(predictions_data), "测试集和预测集长度不一致！"

    return test_data, predictions_data

def load_risks(test_data, predictions_data):


    parsed_preds = []
    parsed_gts = []

    for gt, pred in zip(test_data, predictions_data):
        gt_text = next(m["content"] for m in gt["messages"] if m["role"] == "assistant")
        gt_labels = parse_model_output(gt_text)
        parsed_gts.append(gt_labels)

        pred_labels = parse_model_output(pred["predicted_risks"])
        parsed_preds.append(pred_labels)
    
    return parsed_gts, parsed_preds


if __name__ == "__main__":


    test_data, predictions_data = load_file()
    parsed_gts, parsed_preds = load_risks(test_data, predictions_data)

    # 评估
    precision, recall, f1 = evaluate(parsed_gts, parsed_preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
