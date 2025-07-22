import os
import faiss
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration
API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")
MODEL_NAME = os.getenv("MODEL")

# 1. Read and split the book into chunks

def split_into_chunks(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# 2. Generate embeddings for each chunk

def get_embeddings(chunks_list, embedding_model):
    return embedding_model.encode(chunks_list, show_progress_bar=True)

# 3. Index embeddings with FAISS

def create_faiss_index(embeddings_array):
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)
    return index

# 4. Search for the most relevant chunks

def search_relevant_chunks(question, embedding_model, index, chunks_list, k=5):
    emb_question = embedding_model.encode([question])
    D, I = index.search(emb_question, k)
    return [chunks_list[i] for i in I[0]]

# 5. Build prompt and query DeepSeek

def query_deepseek(context, question):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a professional assistant and can only respond using the provided context."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(DEEPSEEK_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}"

# Example usage with PDF or TXT support
def extract_text_pdf(pdf_path):
    import pdfplumber
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text() or ""
            extracted_text += "\n"
    return extracted_text

if __name__ == "__main__":
    import numpy as np
    file = input("Enter the book file name (.txt or .pdf): ").strip()
    if file.lower().endswith(".pdf"):
        txt_cache = file[:-4] + ".txt"
    elif file.lower().endswith(".txt"):
        txt_cache = file
    else:
        print("Unsupported file format. Use .txt or .pdf")
        exit(1)

    # Extract text only if necessary
    if file.lower().endswith(".pdf") and not os.path.exists(txt_cache):
        print("Extracting text from PDF...")
        file_text = extract_text_pdf(file)
        with open(txt_cache, "w", encoding="utf-8") as f:
            f.write(file_text)
    else:
        with open(txt_cache, "r", encoding="utf-8") as f:
            file_text = f.read()

    # Chunks
    chunks = split_into_chunks(file_text)
    print(f"Generated chunks: {len(chunks)}")

    # Cache files for embeddings and index
    emb_cache = txt_cache + ".embeddings.npy"
    faiss_cache = txt_cache + ".faiss"

    embedding_model_instance = SentenceTransformer("all-MiniLM-L6-v2")

    # If embedding and index files exist, load them
    if os.path.exists(emb_cache) and os.path.exists(faiss_cache):
        print("Loading embeddings and FAISS index from cache...")
        embeddings = np.load(emb_cache)
        faiss_index = faiss.read_index(faiss_cache)
    else:
        print("Generating embeddings and FAISS index...")
        embeddings = get_embeddings(chunks, embedding_model_instance)
        embeddings = np.array(embeddings).astype("float32")
        faiss_index = create_faiss_index(embeddings)
        np.save(emb_cache, embeddings)
        faiss.write_index(faiss_index, faiss_cache)

    # Ask question
    question = input("What topic do you want to ask about?: ")
    relevant_chunks = search_relevant_chunks(question, embedding_model_instance, faiss_index, chunks, k=5)
    context = "\n".join(relevant_chunks)
    answer = query_deepseek(context, question)
    print("\nDeepSeek's Answer:\n", answer)
