from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re


app = FastAPI()


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

def load_model(model_name, bnb_config, adapter_path):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model




# === 加载模型，只加载一次 ===
model_name = "Qwen/Qwen3-8B"
adapter_path = "./data/checkpoints/final_adapter_8B_v3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = load_model(model_name, bnb_config, adapter_path)


# === 定义请求结构体 ===
class DockerfileInput(BaseModel):
    dockerfile_content: str

# === 接口 ===
@app.post("/analyze/")
def analyze_dockerfile(input: DockerfileInput):
    prompt = (
        f"<|im_start|>system\nYou are a professional Dockerfile security audit expert. "
        "For any given Dockerfile, analyze all security risks, and for each risk, provide:\n"
        "- Risk type\n- Risky code snippet\n- Start and end character positions in the file.<|im_end|>\n"
        f"<|im_start|>user\nPlease find all security risks in this Dockerfile and precisely locate the risky code:\n\n"
        f"{input.dockerfile_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predicted_risks = parse_model_output(decoded)


    return {"predicted_risks": predicted_risks}





