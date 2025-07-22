# Chat-PDF

Chat-PDF is a Python tool that allows you to ask questions about the content of a PDF or TXT book using local language models and semantic search. It splits the book into chunks, generates embeddings, indexes them with FAISS, and queries a local QA model using only the most relevant context.

## Features
- Supports PDF and TXT files
- Uses FAISS for fast semantic search
- Uses local HuggingFace models for question answering (no API key required)
- **NEW**: SQLite database storage for improved data organization and performance
- **NEW**: Automatic migration from file-based storage to database
- **NEW**: Better data integrity and relationship management
- Enriches answers by displaying the full sentence from the context containing the extracted answer

## Database Storage

The application now uses an SQLite database instead of separate files for better organization and performance:

### Storage Structure
- **Documents**: Stores PDF metadata, content, and file hashes
- **Chunks**: Stores text chunks with relationships to documents
- **Embeddings**: Stores vector embeddings linked to chunks
- **FAISS Indices**: Stores search indices for fast similarity search

### Advantages
- **Centralized Storage**: Single database file instead of multiple cache files
- **Data Integrity**: Relationships and constraints ensure data consistency
- **Performance**: Faster queries and reduced I/O operations
- **Scalability**: Better handling of multiple documents
- **Migration Support**: Automatic conversion from old file-based storage

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Suppose you have a file called `taxes.pdf` in the project folder. Run:

```bash
python chat-pdf.py
```

You will be prompted:
```
Enter the book file name (.txt or .pdf): taxes.pdf
```

### First Run vs Subsequent Runs

**First run with a new document:**
- Extracts text from PDF/TXT file
- Splits content into chunks
- Generates embeddings using sentence transformers
- Creates FAISS index for similarity search
- Stores everything in SQLite database (`chat_pdf.db`)

**Subsequent runs with the same document:**
- Quickly loads existing data from database
- Skips processing if document hasn't changed
- Much faster startup time

### Migration from File-Based Storage

If you have existing cache files (`.embeddings.npy`, `.faiss`) from previous versions, the application will automatically migrate them to the database on first run.

Then, you will be prompted:
```
What topic do you want to ask about?:
```
Type your question, for example:
```
What are the best algorithms to segment taxpayers?
```
The script will return an answer based only on the most relevant parts of the document using the local model. The answer will be enriched by showing the full sentence from the context where the answer was found.

## Environment Variables
No environment variables or API keys are required. All models are downloaded automatically by HuggingFace Transformers.

## Notes
- The first run on a new document will be slower due to embedding generation and indexing
- Data is stored in `chat_pdf.db` SQLite database
- Database provides better performance and data organization than the previous file-based approach
- Automatic migration from legacy cache files (`.embeddings.npy`, `.faiss`) is supported
- Only the context from your document is used to answer questions
- If you see a warning about symlinks on Windows when using HuggingFace, you can ignore it. To avoid it, enable Windows Developer Mode or run Python as administrator. More info: https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations

## Database Schema

The SQLite database contains the following tables:

- `documents`: Document metadata and content
- `chunks`: Text chunks from processed documents  
- `embeddings`: Vector embeddings for similarity search
- `faiss_indices`: FAISS search indices

## Testing

Run the test suite to verify functionality:

```bash
# Test database functionality
python test_database.py

# Test application integration
python test_app_integration.py

# See database features demo
python demo_database.py
```

## License
MIT
