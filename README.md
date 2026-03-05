# Lightweight Fine-Tuning with PEFT (LoRA)

## Project Goal

This project demonstrates **parameter-efficient fine-tuning (PEFT)** of a pre-trained language model using the Hugging Face `peft` library. Instead of fine-tuning all model parameters (which requires significant compute), PEFT techniques like **LoRA (Low-Rank Adaptation)** allow you to train a tiny fraction of parameters while achieving strong performance — making it practical to fine-tune large models on consumer hardware.

The project covers the full ML pipeline:
1. Load and evaluate a pre-trained foundation model
2. Apply LoRA fine-tuning on a sequence classification task
3. Compare model performance before and after fine-tuning

---

## Tech Stack

- **Model**: GPT-2
- **PEFT Technique**: LoRA (Low-Rank Adaptation)
- **Dataset**: `sms_spam` (Hugging Face) — binary spam/ham classification
- **Frameworks**: PyTorch, Hugging Face `transformers`, `peft`, `evaluate`

---

## Results

| Metric   | Baseline | Fine-Tuned | Improvement |
|----------|----------|------------|-------------|
| Accuracy | 0.8637   | 0.9794     | +0.1157     |
| F1 Score | 0.0000   | 0.9176     | +0.9176     |

The baseline GPT-2 model predicted everything as "ham" (F1 = 0.0), meaning it had no ability to detect spam out of the box. After just **1 epoch of LoRA fine-tuning**, the model achieved **97.9% accuracy** and **0.92 F1 score** — while only training ~0.2% of the total parameters.

---

## Project Structure

```
Lightweight_fine_tuning_AI/
├── LightweightFineTuning.ipynb   # Main project notebook
├── gpt_lora/                     # Saved LoRA adapter weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Guide

### Step 1: Load and Evaluate the Foundation Model

```python
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# Load dataset
dataset = load_dataset("sms_spam", trust_remote_code=True)
split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no default pad token

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id
```

### Step 2: Tokenize the Dataset

```python
def tokenize_function(examples):
    return tokenizer(
        examples["sms"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_eval = tokenized_eval.rename_column("label", "labels")

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

### Step 3: Evaluate Baseline Model

```python
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"]
    return {"accuracy": accuracy, "f1": f1}

baseline_args = TrainingArguments(
    output_dir="./baseline_results",
    per_device_eval_batch_size=8,
    use_cpu=not torch.cuda.is_available(),  # use_cpu replaces deprecated no_cuda
    do_train=False,
    do_eval=True
)

baseline_trainer = Trainer(
    model=model,
    args=baseline_args,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics
)

baseline_trainer.callback_handler.on_train_begin(
    baseline_args, baseline_trainer.state, baseline_trainer.control
)
baseline_results = baseline_trainer.evaluate()
```

### Step 4: Create LoRA Config and Fine-Tune

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS,
    target_modules=["c_attn"]  # GPT-2 attention layer name
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./lora_results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",       # replaces deprecated evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_cpu=not torch.cuda.is_available()
)

lora_trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics
)

lora_trainer.train()
```

### Step 5: Save and Reload the PEFT Model

```python
# Save adapter weights only (much smaller than full model)
lora_model.save_pretrained("/tmp/gpt_lora")
tokenizer.save_pretrained("/tmp/gpt_lora")

# Reload using PEFT-specific class
from peft import AutoPeftModelForSequenceClassification

loaded_lora_model = AutoPeftModelForSequenceClassification.from_pretrained("/tmp/gpt_lora")
loaded_lora_model.config.pad_token_id = tokenizer.pad_token_id
```

### Step 6: Evaluate Fine-Tuned Model and Compare

```python
finetuned_args = TrainingArguments(
    output_dir="./finetuned_results",
    per_device_eval_batch_size=8,
    use_cpu=not torch.cuda.is_available(),
    do_train=False,
    do_eval=True
)

finetuned_trainer = Trainer(
    model=loaded_lora_model,
    args=finetuned_args,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics
)

finetuned_trainer.callback_handler.on_train_begin(
    finetuned_args, finetuned_trainer.state, finetuned_trainer.control
)
finetuned_results = finetuned_trainer.evaluate()
```

---

## Key Takeaways

- LoRA fine-tuning only trained **~0.2% of GPT-2's parameters** yet achieved a massive performance boost
- The adapter weights saved to `gpt_lora/` are only a few MB compared to GPT-2's full 500MB
- `use_cpu` replaces the deprecated `no_cuda` argument in newer versions of `transformers`
- `eval_strategy` replaces the deprecated `evaluation_strategy` argument

---

## References

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
