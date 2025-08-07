import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
from tqdm import tqdm
from datasets import load_dataset


def load_model_with_adapter(model_name,bnb_config,adapter_path):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config
    )
    ## 加载训练的权重
    model = PeftModel.from_pretrained(model, adapter_path)
    ## 开启推理模式
    model.eval()

    return model


def eval_process(test_dataset):

    results = []

    for sample in tqdm(test_dataset):
        
        system_content = next(m["content"] for m in sample["messages"] if m["role"] == "system")
        user_content = next(m["content"] for m in sample["messages"] if m["role"] == "user")

        prompt = (
            f"<|im_start|>system\n{system_content}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,   # 完全去除随机性
                do_sample=False    # 关闭采样，用贪婪搜索        
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "original_file": user_content,
            "predicted_risks": decoded
        })
    return results


if __name__ == "__main__":

    model_name = "Qwen/Qwen3-8B"
    adapter_path = "/root/autodl-tmp/data/checkpoints/final_adapter_8B_v3"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = load_model_with_adapter(model_name,bnb_config,adapter_path) 

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    test_dataset = load_dataset("json", data_files="../data/evaluation_data/qwen_test.jsonl")["train"]

    results = eval_process(test_dataset)

    with open("./predictions/test_predictions.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Inference has been finished, a total of", len(results), "samples processed")
