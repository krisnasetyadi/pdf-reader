# Format data untuk fine-tuning
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

qa_dataset = [
    {
        "context": "Dokumen lengkap tentang subjek tertentu...",
        "question": "Pertanyaan tentang dokumen?",
        "answer": "Jawaban yang diinginkan"
    },
    # ... lebih banyak contoh
]


# Load model dan tokenizer
model_name = "cahya/bert-base-indonesian-1.5G"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Preprocess data


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Implementasi lengkap untuk extract answer positions
    # ... (disini perlu logic untuk menemukan posisi jawaban dalam teks)

    return inputs


# Training arguments
training_args = TrainingArguments(
    output_dir="./qa-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Improved prompt template
advanced_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.

INSTRUKSI:
1. Jawablah pertanyaan hanya berdasarkan informasi yang ada dalam konteks
2. Jika informasi tidak cukup, jelaskan bahwa informasi tidak ditemukan
3. Gunakan bahasa yang jelas dan mudah dimengerti
4. Untuk pertanyaan definisi, berikan penjelasan singkat dan padat

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""
)


# Script untuk evaluasi model
def evaluate_qa_system(test_dataset):
    results = []
    for item in test_dataset:
        start_time = time.time()
        answer = process_query(item["question"], item["context"])
        end_time = time.time()

        # Calculate metrics
        similarity = calculate_similarity(answer, item["expected_answer"])
        results.append({
            "question": item["question"],
            "expected": item["expected_answer"],
            "actual": answer,
            "similarity": similarity,
            "processing_time": end_time - start_time
        })
    return results
