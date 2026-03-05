# Lightweight_fine_tuning_AI

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
- **Dataset**: Hugging Face `datasets` library
- **Frameworks**: PyTorch, Hugging Face `transformers`, `peft`, `evaluate`

---

## Project Structure

```
Lightweight_fine_tuning_AI/
├── notebook.ipynb        # Main project notebook
├── gpt_lora/             # Saved LoRA adapter weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Step-by-Step Guide

### Prerequisites

Make sure you have Python 3.8+ installed. Then install dependencies:

```bash
pip install -r requirements.txt
```

---

### Step 1: Load and Evaluate the Foundation Model

In `notebook.ipynb`, the pre-trained GPT-2 model is loaded for sequence classification:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

The model is evaluated on a dataset to record its **baseline performance** (accuracy/F1).

---

### Step 2: Create a PEFT Config (LoRA)

A LoRA config is created to define the adapter hyperparameters:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="SEQ_CLS"
)

lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()
```

This converts the model into a PEFT model where only the LoRA adapter parameters are trained (~0.2% of total parameters).

---

### Step 3: Fine-Tune the PEFT Model

The LoRA model is trained using the Hugging Face `Trainer`:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

---

### Step 4: Save the Trained Adapter

Only the adapter weights are saved (much smaller than the full model):

```python
lora_model.save_pretrained("gpt_lora")
```

---

### Step 5: Load and Evaluate the Fine-Tuned Model

The saved PEFT model is loaded using the PEFT-specific class:

```python
from peft import AutoPeftModelForSequenceClassification

loaded_model = AutoPeftModelForSequenceClassification.from_pretrained("gpt_lora")
```

The fine-tuned model is then evaluated and its performance is compared to the original baseline.

---

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Base GPT-2 | TBD | TBD |
| LoRA Fine-Tuned GPT-2 | TBD | TBD |

> Results will be populated after running the notebook.

---

## References

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)