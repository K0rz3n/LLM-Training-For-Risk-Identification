import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, concatenate_datasets
import json
from transformers import DataCollatorWithPadding, TrainerCallback
import wandb

class DataCollatorForCausalLMWithAssistantMask:
    """
    - 动态padding input_ids/attention_mask
    - labels只在Assistant段保留,其他地方是-100
    - 遇到一个样本label全是-100时,强制保留少量token,避免loss为NaN
    """

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.default_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt"
        )
        self.assistant_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.assistant_role_token_id = self.tokenizer.convert_tokens_to_ids("assistant")

    def __call__(self, features):
        # 提取 labels
        labels = [feature.pop("labels") for feature in features]

        # collate其他input_ids和attention_mask
        batch = self.default_collator(features)

        max_len = batch["input_ids"].shape[1]
        padded_labels = []

        for input_ids, label in zip(batch["input_ids"], labels):
            # Padding
            padding_len = max_len - len(label)
            label = label + [-100] * padding_len

            keep_mask = [False] * len(label)
            found_assistant = False

            for i in range(len(label) - 1):
                if (input_ids[i] == self.assistant_start_token_id and input_ids[i + 1] == self.assistant_role_token_id):
                    found_assistant = True
                if found_assistant:
                    keep_mask[i] = True

            # 根据keep_mask生成最终label
            filtered_label = [
                l if keep else -100
                for l, keep in zip(label, keep_mask)
            ]

            # fallback：如果全是-100，至少保留原label中的一个正常token
            if all(l == -100 for l in filtered_label):
                for idx, l in enumerate(label):
                    if l != -100:
                        filtered_label[idx] = l
                        break  # 只保留一个，够了

            padded_labels.append(filtered_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch



def chatml_preprocess(example):
    prompt = ""
    messages = example["messages"]

    # Qwen3 格式
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False  # 显式关闭 Qwen3 的“深度思考”模式
    )
    
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding=False,       # 单条不要padding
        return_tensors=None  # 返回list[int]，不要tensor
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    assert isinstance(input_ids, list) and all(isinstance(x, int) for x in input_ids), "input_ids不是list of int!"
    assert isinstance(attention_mask, list) and all(isinstance(x, int) for x in attention_mask), "attention_mask不是list of int!"

    # labels = input_ids.copy() 保证是list
    labels = input_ids.copy()
    
    assert isinstance(labels, list) and all(isinstance(x, int) for x in labels), "labels不是list of int!"

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def load_config():
        # 载入配置
    with open("./configs/config_lora.json") as f:
        lora_cfg = json.load(f)
    with open("./configs/config_train.json") as f:
        train_cfg = json.load(f)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    return lora_cfg,train_cfg,bnb_config

def load_model(model_name,lora_cfg,bnb_config,adapter_path=None):


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )

    # 进一步学习
    # model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    
  
    # mac 本地训练使用: 不使用GPU，使用CPU
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     trust_remote_code=True,
    #     device_map=None,          # 不要"auto"，取消设备映射
    #     low_cpu_mem_usage=False,   # 强制完整加载到CPU内存
    #     use_cache=False
    # )

    # 初次训练
    lora_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, lora_config)


    # 通用写法启用 gradient checkpointing
    model.base_model.model.gradient_checkpointing_enable()

    ## 让 LoRA的参数开启 requires_grad，否则会出现 backword 时候 grad_fn 找不到的情况
    model.enable_input_require_grads()

    # assert isinstance(model, PeftModel), "模型没有正确包裹为 PeftModel！"

    ## 打印可训练的参数比例
    model.print_trainable_parameters()

    return model


def get_trainer(train_cfg,model,train_dataset,val_dataset,tokenizer,data_collator):

    training_args = TrainingArguments(
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="qwen3-8b-qlora",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_strategy="steps",
        save_strategy="steps",
        **train_cfg
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ]

    )

    return trainer

if __name__ == "__main__":

    # 进一步学习
    # adapter_path = "/root/autodl-tmp/data/checkpoints/final_adapter"
    
    lora_cfg,train_cfg,bnb_config = load_config()
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_dataset("json", data_files="../data/trainning_data/qwen_train.jsonl")["train"]
    val_dataset = load_dataset("json", data_files="../data/trainning_data/qwen_val.jsonl")["train"]

    train_dataset = train_dataset.map(chatml_preprocess, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(chatml_preprocess, remove_columns=val_dataset.column_names)


    data_collator = DataCollatorForCausalLMWithAssistantMask(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    # 进一步学习
    # model  = load_model(model_name,lora_cfg,bnb_config,adapter_path)
    model  = load_model(model_name,lora_cfg,bnb_config)

    trainer = get_trainer(train_cfg,model,train_dataset,val_dataset,tokenizer,data_collator)

    # 中断后继续训练
    #trainer.train(resume_from_checkpoint=True)

    trainer.train()

    model.save_pretrained("/root/autodl-tmp/data/checkpoints/final_adapter_8B_v3", save_adapter=True)
    tokenizer.save_pretrained("/root/autodl-tmp/data/checkpoints/final_adapter_8B_v3")
    wandb.finish()
