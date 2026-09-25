# Don't Agent fine-tuning
# Base: Qwen2.5-Coder-1.5B | Method: LoRA via Unsloth | Target: Colab free GPU (T4)
# Run in Colab:  !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#                 then: !python finetune.py

import os, json, torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

LOCAL_DIR = "/content/qwen25-coder-1.5b"
MODEL_ID = LOCAL_DIR if os.path.isdir(LOCAL_DIR) else "Qwen/Qwen2.5-Coder-1.5B"
MAX_SEQ = 2048
MAX_STEPS = 250          # raise for longer training (e.g. 500)
TRAIN_ROWS = 4000        # cap of code rows (free Colab speed)
OUT_DIR = "/content/dont-agent-lora"

SYSTEM = (
    "You are Don't Agent, an expert AI programming agent. "
    "You write clean, correct code, explain your reasoning step by step, "
    "and break complex tasks into clear actionable steps. "
    "You are helpful, precise, and honest about limitations."
)

def to_chatml(instruction, inp, output):
    user = instruction + (f"\n{inp}" if inp else "")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]

def load_training_data():
    rows = []
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for r in ds:
            rows.append(to_chatml(r["instruction"], r.get("input", ""), r["output"]))
        print(f"code dataset rows: {len(rows)}")
    except Exception as e:
        print("primary code dataset failed, using fallback:", e)
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        for r in ds:
            rows.append(to_chatml(r["instruction"], r.get("input", ""), r["output"]))
    rows = rows[:TRAIN_ROWS]
    # blend the Don't Agent persona file if present
    for cand in ("/content/persona.jsonl", "persona.jsonl"):
        if os.path.exists(cand):
            with open(cand) as f:
                for line in f:
                    ex = json.loads(line)
                    rows.append([
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": ex["user"]},
                        {"role": "assistant", "content": ex["assistant"]},
                    ])
            print(f"blended persona file: {cand}")
            break
    return Dataset.from_list([{"messages": r} for r in rows])

print("loading base model:", MODEL_ID)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

data = load_training_data()

def fmt(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

data = data.map(fmt)
print("final training rows:", len(data))

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="/content/dont-agent-out",
    ),
)

trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("DONE ->", OUT_DIR)
