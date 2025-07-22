# Chat-PDF

Chat-PDF is a Python tool that allows you to ask questions about the content of a PDF or TXT book using local language models and semantic search. It splits the book into chunks, generates embeddings, indexes them with FAISS, and queries a local QA model using only the most relevant context.

## Features
- Supports PDF and TXT files
- Uses FAISS for fast semantic search
- Uses local HuggingFace models for question answering (no API key required)
- Caches embeddings and index for fast repeated queries
- Enriches answers by displaying the full sentence from the context containing the extracted answer

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
If it's the first run, the script will extract the text and generate embeddings and an index (this may take a while for large files).

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
- The first run on a new file will be slower due to embedding and indexing.
- Embeddings and index are cached for each TXT file.
- Only the context from your document is used to answer questions.
- If you see a warning about symlinks on Windows when using HuggingFace, you can ignore it. To avoid it, enable Windows Developer Mode or run Python as administrator. More info: https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations

## License
MIT
