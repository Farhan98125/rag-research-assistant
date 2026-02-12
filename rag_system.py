"""
Agentic RAG Research Assistant
Implements Retrieval-Augmented Generation using
MiniLM embeddings + FAISS vector search.
"""

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pypdf import PdfReader


# -------------------------
# 1. Load PDF (First 4 Pages)
# -------------------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages[:4]:
        text += page.extract_text()

    # Clean text
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


# -------------------------
# 2. Chunking
# -------------------------
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# -------------------------
# 3. Build Vector Store
# -------------------------
def build_vector_store(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return model, index


# -------------------------
# 4. Retrieval
# -------------------------
def retrieve(query, model, index, chunks, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved = [chunks[i] for i in indices[0]]
    return "\n\n".join(retrieved)


# -------------------------
# 5. Generation
# -------------------------
def generate_answer(query, context):
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    prompt = f"""
Context:
{context}

Question:
{query}

Answer clearly:
"""

    output = generator(prompt, max_new_tokens=150)[0]["generated_text"]
    return output


# -------------------------
# 6. Run Full RAG
# -------------------------
def run_rag(pdf_path, query):

    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    embedding_model, index = build_vector_store(chunks)

    context = retrieve(query, embedding_model, index, chunks)
    answer = generate_answer(query, context)

    print("\n===== RETRIEVED CONTEXT =====")
    print(context[:500])

    print("\n===== FINAL ANSWER =====")
    print(answer)


# -------------------------
# Example Run
# -------------------------
if __name__ == "__main__":
    run_rag("paper.pdf", "What problem does the paper solve?")
