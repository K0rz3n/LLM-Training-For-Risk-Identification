import json
import pandas as pd

def format_to_chatml(file_content, risks):
    sys_prompt = (
        "You are a professional Dockerfile security audit expert. "
        "For any given Dockerfile, analyze all security risks, and for each risk, provide:\n"
        "- Risk type\n"
        "- Risky code snippet\n"
        "- Start and end character positions in the file."
    )

    user_prompt = (
        "Please find all security risks in this Dockerfile and precisely locate the risky code:\n\n"
        + file_content.strip()
    )

    if risks:
        risk_blocks = []
        for idx, risk in enumerate(risks, 1):
            block = (
                f"{idx}. Risk type: {risk['label']}\n"
                f"Snippet: '{risk['text']}'\n"
                f"Position: {risk['start']}-{risk['end']}"
            )
            risk_blocks.append(block)
        assistant_response = "Detected security risks:\n\n" + "\n\n".join(risk_blocks)
    else:
        assistant_response = "No obvious security vulnerabilities were detected."

    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }

# 转换并保存成 JSONL
def convert_df_to_qwen_jsonl(df, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = format_to_chatml(row["file_content"], row["risk_labels"])
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")



df_train = pd.read_json("/Users/k0rz3n/projects/individualProject/QWEN_Classfier/data/splits/third/train_balanced.json")
df_val = pd.read_json("/Users/k0rz3n/projects/individualProject/QWEN_Classfier/data/splits/third/val.json")
df_test = pd.read_json("/Users/k0rz3n/projects/individualProject/QWEN_Classfier/data/splits/third/test.json")

# 使用
convert_df_to_qwen_jsonl(df_train, "qwen_train.jsonl")
convert_df_to_qwen_jsonl(df_val, "qwen_val.jsonl")
convert_df_to_qwen_jsonl(df_test, "qwen_test.jsonl")