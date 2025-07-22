# Chat-PDF

Chat-PDF is a Python tool that allows you to ask questions about the content of a PDF or TXT book using LLMs and semantic search. It splits the book into chunks, generates embeddings, indexes them with FAISS, and queries an LLM (DeepSeek) using only the most relevant context.

## Features
- Supports PDF and TXT files
- Uses FAISS for fast semantic search
- Integrates with DeepSeek LLM API
- Caches embeddings and index for fast repeated queries

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the example environment file and set your API key:
   ```bash
   cp .env.example .env
   # Edit .env and set your DEEPSEEK_API_KEY
   ```

## Usage Example

Suppose you have a file called `taxes.pdf` in the project folder. Run:

```bash
python chat-pdf.py
```

You will be prompted:
```
Enter the book file name (.txt or .pdf): taxes.pdf
```
If it's the first run, the script will extract the text and generate embeddings and an index (this may take a while for large files).

Then, you will be prompted:
```
What topic do you want to ask about?:
```
Type your question, for example:
```
What are the best algorithms to segment taxpayers?
```
The script will return an answer from DeepSeek based only on the most relevant parts of the document.

## Environment Variables
Set these in your `.env` file:
```
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_URL=https://api.deepseek.com/chat/completions
MODEL=deepseek-reasoner
```

## Notes
- The first run on a new file will be slower due to embedding and indexing.
- Embeddings and index are cached for each TXT file.
- Only the context from your document is used to answer questions.

## License
MIT
