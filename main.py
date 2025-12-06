import os
import chromadb
from chromadb.utils import embedding_functions
import ollama

# ======================================
# Setup Chroma client and collection
# ======================================
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"

# We'll define a custom embedding function using Ollama below,
# since Ollama doesn’t natively integrate with Chroma yet.
def get_ollama_embedding(text, model="mistral-openorca"):
    """Generate embeddings for text using Ollama."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# ======================================
# Load documents
# ======================================
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# ======================================
# Split text into chunks
# ======================================
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# ======================================
# Load and chunk documents
# ======================================
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})


# ======================================
# Generate embeddings and upsert to Chroma
# ======================================
collection = chroma_client.get_or_create_collection(name=collection_name)

for doc in chunked_documents:
    print(f"==== Generating embeddings for {doc['id']} ====")
    embedding = get_ollama_embedding(doc["text"])
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]], embeddings=[embedding])


# ======================================
# Query function
# ======================================
def query_documents(question, n_results=2):
    print("==== Querying relevant chunks ====")
    # Generate embedding for question
    query_embedding = get_ollama_embedding(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return relevant_chunks


# ======================================
# Generate answer using Ollama
# ======================================
def generate_response(question, relevant_chunks, model="mistral-openorca"):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following context to answer the question concisely. "
        "If you don't know the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ])

    answer = response["message"]["content"]
    return answer


# ======================================
# Example query
# ======================================
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print("\n=== ANSWER ===")
print(answer)
