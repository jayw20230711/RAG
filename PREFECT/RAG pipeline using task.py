from prefect import task

# Load Documents
@task(retries=3)
def load_documents(path: str):
    from pathlib import Path

    docs = []
    for file in Path(path).glob("*.txt"):
        docs.append(file.read_text(encoding="utf-8"))

    return docs

# Chunk Documents
@task
def chunk_documents(docs, chunk_size=500, overlap=50):
    chunks = []
    for doc in docs:
        for i in range(0, len(doc), chunk_size - overlap):
            chunks.append(doc[i:i + chunk_size])
    return chunks

# Embed Chunks
@task(tags=["embeddings", "cpu"])
def embed_chunks(chunks):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Build Vector Store
@task
def build_vector_store(embeddings):
    import faiss
    import numpy as np

    #dim = embeddings.shape[1]
    #index = faiss.IndexFlatL2(dim)
    #index.add(np.array(embeddings))

    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Vectors in index:", index.ntotal)
    return index

# Retrieve Context
@task
def retrieve(query, index, chunks, k=3):
    import numpy as np

    query_embedding = embed_chunks.fn([query])[0]
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]

#Generate Answers
@task(tags=["generation", "llm"])
def generate_answer(query, contexts):
    from transformers import pipeline

    generator = pipeline(
        "text-generation",
        model="microsoft/phi-3-mini-4k-instruct",
        device_map="auto"
    )

    prompt = f"""
    Use the context below to answer the question.

    Context:
    {''.join(contexts)}

    Question:
    {query}
    """

    output = generator(prompt, max_new_tokens=200)
    return output[0]["generated_text"]

# Define the Flow
from prefect import flow

@flow(name="rag-pipeline")
def rag_pipeline(doc_path: str, question: str):
    docs = load_documents(doc_path)
    chunks = chunk_documents(docs)
    embeddings = embed_chunks(chunks)
    index = build_vector_store(embeddings)

    contexts = retrieve(question, index, chunks)
    answer = generate_answer(question, contexts)

    print(answer)

rag_pipeline("docs/", "What does this document say about LangChain?")

print("FLOW FINISHED — waiting before exit")
import time
time.sleep(10)