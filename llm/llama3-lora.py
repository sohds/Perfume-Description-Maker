# 필수 라이브러리
# !pip install datasets transformers peft accelerate bitsandbytes

import torch
import transformers
import pandas as pd
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq

# 1. JSONL 파일 로드
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# 데이터 로드
df = load_jsonl('../preprocess/final/fine_tune_dataset.jsonl')
dataset = Dataset.from_pandas(df)

# 2. 프롬프트 템플릿 설정
def create_prompt_template(row):
    instruction = row['prompt']

    return {
        "instruction": instruction,
        "response": row['completion'],
        "text": f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\nInstruction: {instruction}\nResponse: {row['completion']}"
    }

processed_dataset = dataset.map(create_prompt_template)

# 3. 토크나이저와 모델 로드
model_id = "OpenBuddy/openbuddy-llama3.2-1b-v23.1-131k"  # OpenBuddy 1B 모델
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
model.resize_token_embeddings(len(tokenizer))

# 4. 데이터 전처리
MAX_LENGTH = 512

def preprocess_batch(batch):
    model_inputs = tokenizer(
        batch["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )
    model_inputs["labels"] = model_inputs['input_ids'].copy()
    return model_inputs

encoded_dataset = processed_dataset.map(
    preprocess_batch,
    batched=True,
    remove_columns=processed_dataset.column_names
)

# 데이터셋 분할
split_dataset = encoded_dataset.train_test_split(test_size=0.1, seed=42)

# 5. LoRA 설정
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# 모델 준비
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="perfume_description_generator",
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# 7. 트레이너 설정 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
)

# 8. 학습 실행
model.config.use_cache = False
trainer.train()

# 9. 모델 저장
trainer.model.save_pretrained("perfume_description_generator")

