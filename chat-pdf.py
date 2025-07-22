import os
import faiss
import tempfile
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import pipeline
from database import ChatPDFDatabase

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
        chunk = " ".join(words[i:i + max_words])
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


# Helper function to save FAISS index to bytes
def faiss_index_to_bytes(index):
    with tempfile.NamedTemporaryFile() as temp_file:
        faiss.write_index(index, temp_file.name)
        temp_file.seek(0)
        return temp_file.read()


# Helper function to load FAISS index from bytes
def faiss_index_from_bytes(index_bytes):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(index_bytes)
        temp_file.flush()
        index = faiss.read_index(temp_file.name)
        os.unlink(temp_file.name)
        return index


# 4. Search for the most relevant chunks
def search_relevant_chunks(question, embedding_model, index, chunks_list, k=5):
    emb_question = embedding_model.encode([question])
    D, I = index.search(emb_question, k)
    return [chunks_list[i] for i in I[0]]


# 5. Build prompt and query local model
def query_local_qa(context, question, qa_pipeline=None):
    if qa_pipeline is None:
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=context)
    return result["answer"]


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

    # Initialize database
    db = ChatPDFDatabase()

    file = input("Enter the book file name (.txt or .pdf): ").strip()
    if file.lower().endswith(".pdf"):
        txt_cache = file[:-4] + ".txt"
    elif file.lower().endswith(".txt"):
        txt_cache = file
    else:
        print("Unsupported file format. Use .txt or .pdf")
        exit(1)

    # Check if document is already processed in database
    if db.document_exists(file):
        print("Loading document from database...")
        document_id, file_text, chunks = db.get_document_by_filename(file)
    else:
        # Extract text only if necessary
        if file.lower().endswith(".pdf"):
            print("Extracting text from PDF...")
            file_text = extract_text_pdf(file)
            
            # Save to temporary txt file for compatibility if needed
            if not os.path.exists(txt_cache):
                with open(txt_cache, "w", encoding="utf-8") as f:
                    f.write(file_text)
        else:
            if not os.path.exists(file):
                print(f"File {file} not found.")
                exit(1)
            with open(file, "r", encoding="utf-8") as f:
                file_text = f.read()

        # Generate chunks
        chunks = split_into_chunks(file_text)
        print(f"Generated chunks: {len(chunks)}")
        
        # Store document in database
        document_id = db.store_document(file, file_text, chunks)

    print(f"Document loaded with {len(chunks)} chunks")

    embedding_model_instance = SentenceTransformer("all-MiniLM-L6-v2")

    # Check if embeddings and FAISS index exist in database
    embeddings = db.get_embeddings(document_id)
    faiss_index_bytes = db.get_faiss_index(document_id)

    if embeddings is not None and faiss_index_bytes is not None:
        print("Loading embeddings and FAISS index from database...")
        faiss_index = faiss_index_from_bytes(faiss_index_bytes)
    else:
        print("Generating embeddings and FAISS index...")
        
        # Try to migrate from existing files first
        if not embeddings and db.migrate_from_files(file):
            print("Migrated existing file-based data to database...")
            embeddings = db.get_embeddings(document_id)
            faiss_index_bytes = db.get_faiss_index(document_id)
            if faiss_index_bytes:
                faiss_index = faiss_index_from_bytes(faiss_index_bytes)
            else:
                # Generate new index
                embeddings_array = embeddings.astype("float32")
                faiss_index = create_faiss_index(embeddings_array)
                index_bytes = faiss_index_to_bytes(faiss_index)
                db.store_faiss_index(document_id, index_bytes, embeddings_array.shape[1])
        else:
            # Generate new embeddings and index
            embeddings = get_embeddings(chunks, embedding_model_instance)
            embeddings = np.array(embeddings).astype("float32")
            faiss_index = create_faiss_index(embeddings)
            
            # Store in database
            db.store_embeddings(document_id, embeddings)
            index_bytes = faiss_index_to_bytes(faiss_index)
            db.store_faiss_index(document_id, index_bytes, embeddings.shape[1])

    # Ask question
    question = input("What topic do you want to ask about?: ")
    relevant_chunks = search_relevant_chunks(question, embedding_model_instance, faiss_index, chunks, k=20)
    context = "\n".join(relevant_chunks)
    print("Running local QA model...")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    answer = query_local_qa(context, question, qa_pipeline)
    # Search for the full sentence containing the answer
    import re


    def find_full_sentence(context, answer):
        # Search for the sentence containing the answer
        pattern = r'([^.\n]*?\b' + re.escape(answer) + r'\b[^.\n]*[.\n])'
        matches = re.findall(pattern, context, flags=re.IGNORECASE)
        if matches:
            # Show the longest sentence found
            return max(matches, key=len).strip()
        return answer


    enriched = find_full_sentence(context, answer)
    print("\nLocal Model's Answer (enriched):\n", enriched)
