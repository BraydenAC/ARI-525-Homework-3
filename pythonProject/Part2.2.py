import pandas as pd
import transformers
from datasets import Dataset
from sklearn.model_selection import train_test_split
# from sympy import evaluate
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, Trainer, \
    AutoModelForSequenceClassification
from transformers import TrainingArguments
import torch
import evaluate
import numpy as np

df = pd.read_parquet("hf://datasets/ucirvine/sms_spam/plain_text/train-00000-of-00001.parquet")
#train-test-split(from ChatGPT)
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dev_test = dataset["test"].train_test_split(test_size=0.5, seed=42)
# Load models directly
model_name = "FacebookAI/roberta-base"
# model_name = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("Model loaded...")

def tokenize(examples):
    output = tokenizer(examples["sms"], padding="max_length", truncation=True, max_length=128)
    return {
        "input_ids": output["input_ids"],
        "attention_mask": output["attention_mask"],
        "labels": examples["label"]
    }
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["label"])
train = dataset["train"]
dev = dev_test["train"]
test = dev_test["test"]
print("Dataset tokenized...")

#Following code snippet from fine tune guide
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="model_save",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    fp16=True,
    gradient_accumulation_steps=2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=dev,
    compute_metrics=compute_metrics,
)
print("Trainer Initialized...")
trainer.train()
# print(train("input"))

print("Trainer complete...")