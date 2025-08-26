import csv
from processor import processor


def build_qa_dataset(collection_id, output_file="qa_dataset.csv", max_samples=100):
    """
    Membuat dataset Q&A otomatis dari PDF chunks dalam koleksi tertentu.
    """
    # 1. Ambil vector store (FAISS)
    vs = processor.get_vector_store(collection_id)
    if not vs:
        raise ValueError(f"Collection {collection_id} not found")

    # 2. Ambil semua dokumen (chunks)
    docs = vs.similarity_search(
        "dummy", k=vs.index.ntotal)  # ambil semua chunk
    print(f"Loaded {len(docs)} chunks from collection {collection_id}")

    qa_dataset = []

    for d in docs[:max_samples]:
        context = d.page_content

        # 3. Generate pertanyaan dari context
        q = processor.llm.predict(
            f"Buatkan 1 pertanyaan singkat dan jelas dari teks berikut:\n\n{context}"
        )

        # 4. Generate jawaban berdasarkan context
        a = processor.llm.predict(
            f"Jawab pertanyaan berikut hanya dengan konteks yang diberikan.\n\nKonteks: {context}\nPertanyaan: {q}\nJawaban:"
        )

        qa_dataset.append({"context": context, "question": q, "answer": a})

    # 5. Simpan ke CSV
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["context", "question", "answer"])
        writer.writeheader()
        writer.writerows(qa_dataset)

    print(f"Dataset saved to {output_file} with {len(qa_dataset)} samples")


# Contoh penggunaan
if __name__ == "__main__":
    build_qa_dataset(
        collection_id="15bdca2e-b50b-46df-8115-f25aba1279a1",
        output_file="qa_dataset.csv",
        max_samples=50
    )
