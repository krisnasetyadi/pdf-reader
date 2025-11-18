# fine_tune_qa_t5.py
import os
import re
import string
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    T5TokenizerFast, T5ForConditionalGeneration,
    DataCollatorForSeq2Seq, TrainingArguments, Trainer
)
from evaluate import load
import torch
import matplotlib.pyplot as plt

# -----------------------------
# 1) Config
# -----------------------------
MODEL_NAME = "t5-small"            # kecil & cepat; bisa ganti t5-base
DATA_PATH = "data/qa_dataset.csv"  # CSV: context,question,answer
OUTPUT_DIR = "outputs/t5_qa_finetune"
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 64
NUM_EPOCHS = 100                   # sesuai permintaan dosen
BATCH_SIZE = 4                     # kecil supaya muat di GPU/CPU
GRAD_ACCUM = 4                     # 4 * 4 = "virtual" batch 16
LEARNING_RATE = 3e-4
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 2) Load & split dataset
# -----------------------------
df = pd.read_csv(DATA_PATH).dropna(subset=["context", "question", "answer"])
# Optional: dedup
df = df.drop_duplicates(
    subset=["context", "question", "answer"]).reset_index(drop=True)

train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
raw_ds = DatasetDict({"train": train_ds, "validation": val_ds})

# -----------------------------
# 3) Tokenizer & formatting
# -----------------------------
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)


def build_source(row):
    # Prompt gaya instruksi: bisa disesuaikan dengan prompt_template di proyekmu
    return f"Jawab singkat dan akurat berdasarkan konteks.\nKonteks: {row['context']}\nPertanyaan: {row['question']}\nJawaban:"


def preprocess_func(examples):
    inputs = [build_source(r) for r in examples]
    model_inputs = tokenizer(
        inputs, max_length=MAX_SOURCE_LEN, truncation=True)

    labels = tokenizer(
        examples["answer"],
        max_length=MAX_TARGET_LEN,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed = raw_ds.map(
    lambda x: preprocess_func(
        {"context": x["context"], "question": x["question"], "answer": x["answer"]}),
    batched=True,
    remove_columns=raw_ds["train"].column_names
)

# -----------------------------
# 4) Model & collator
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# -----------------------------
# 5) Metrics (EM, token-F1/Prec/Rec, ROUGE-L)
# -----------------------------
rouge = load("rouge")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)  # remove articles (EN)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


def precision_recall_f1(pred_tokens: List[str], gold_tokens: List[str]):
    pred_set, gold_set = set(pred_tokens), set(gold_tokens)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # decode predictions
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # replace -100 with pad_token_id before decoding labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    gold_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # EM, token P/R/F1
    ems, ps, rs, f1s = [], [], [], []
    for p, g in zip(pred_str, gold_str):
        ems.append(exact_match(p, g))
        p_tokens = normalize_text(p).split()
        g_tokens = normalize_text(g).split()
        prec, rec, f1 = precision_recall_f1(p_tokens, g_tokens)
        ps.append(prec)
        rs.append(rec)
        f1s.append(f1)

    # ROUGE-L
    rouge_scores = rouge.compute(predictions=pred_str, references=gold_str)
    metrics = {
        "exact_match": float(np.mean(ems)),
        "precision": float(np.mean(ps)),
        "recall": float(np.mean(rs)),
        "f1": float(np.mean(f1s)),
        "rougeL": float(rouge_scores["rougeL"])
    }
    return metrics


# -----------------------------
# 6) Training args
# -----------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    report_to=[]  # matikan W&B dsb
)

# -----------------------------
# 7) Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=processed["train"],
    eval_dataset=processed["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# -----------------------------
# 8) Train & Save
# -----------------------------
train_result = trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Simpan history metrics ke CSV
logs = pd.DataFrame(trainer.state.log_history)
logs.to_csv(os.path.join(OUTPUT_DIR, "training_logs.csv"), index=False)

# Plot Loss vs Steps
loss_df = logs[logs["loss"].notna()][["step", "loss"]]
if not loss_df.empty:
    plt.figure()
    plt.plot(loss_df["step"], loss_df["loss"])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"),
                bbox_inches="tight")

# Plot EM/F1 vs Epoch
eval_df = logs[logs["eval_loss"].notna()]
if not eval_df.empty:
    for m in ["eval_exact_match", "eval_f1", "eval_precision", "eval_recall",
              "eval_rougeL"]:
        if m in eval_df.columns:
            plt.figure()
            plt.plot(eval_df["epoch"], eval_df[m])
            plt.xlabel("Epoch")
            plt.ylabel(m)
            plt.title(m)
            plt.savefig(os.path.join(
                OUTPUT_DIR, f"{m}.png"), bbox_inches="tight")

print("✅ Selesai. Model & metric tersimpan di:", OUTPUT_DIR)
