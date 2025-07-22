import sqlite3
import hashlib
import json
import pickle
import numpy as np
from datetime import datetime
import os
from typing import List, Optional, Tuple, Any


class ChatPDFDatabase:
    """Database manager for storing PDF documents, chunks, embeddings, and FAISS indices."""
    
    def __init__(self, db_path: str = "chat_pdf.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                file_hash TEXT NOT NULL,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content_text TEXT NOT NULL
            )
        ''')
        
        # Chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id),
                UNIQUE(document_id, chunk_index)
            )
        ''')
        
        # Embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL,
                embedding_vector BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks (id),
                UNIQUE(chunk_id)
            )
        ''')
        
        # FAISS indices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faiss_indices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                index_data BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id),
                UNIQUE(document_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_file_hash(self, filename: str) -> str:
        """Generate SHA256 hash of file content."""
        if not os.path.exists(filename):
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(filename, "rb") as f:
            # Read and update hash in chunks of 64K
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def document_exists(self, filename: str) -> bool:
        """Check if document exists and is up to date."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current file hash
        current_hash = self._get_file_hash(filename)
        if not current_hash:
            conn.close()
            return False
        
        # Check if document exists with same hash
        cursor.execute(
            "SELECT id FROM documents WHERE filename = ? AND file_hash = ?",
            (filename, current_hash)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    def store_document(self, filename: str, content_text: str, chunks: List[str]) -> int:
        """Store document and its chunks in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        file_hash = self._get_file_hash(filename)
        
        try:
            # Insert or update document
            cursor.execute('''
                INSERT OR REPLACE INTO documents (filename, file_hash, content_text, processed_date)
                VALUES (?, ?, ?, ?)
            ''', (filename, file_hash, content_text, datetime.now()))
            
            document_id = cursor.lastrowid
            
            # Clear existing chunks for this document
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            
            # Insert chunks
            for i, chunk in enumerate(chunks):
                word_count = len(chunk.split())
                cursor.execute('''
                    INSERT INTO chunks (document_id, chunk_index, content, word_count)
                    VALUES (?, ?, ?, ?)
                ''', (document_id, i, chunk, word_count))
            
            conn.commit()
            return document_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_document_by_filename(self, filename: str) -> Optional[Tuple[int, str, List[str]]]:
        """Get document and chunks by filename."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get document
        cursor.execute(
            "SELECT id, content_text FROM documents WHERE filename = ?",
            (filename,)
        )
        doc_result = cursor.fetchone()
        
        if not doc_result:
            conn.close()
            return None
        
        document_id, content_text = doc_result
        
        # Get chunks
        cursor.execute(
            "SELECT content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,)
        )
        chunks = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return document_id, content_text, chunks
    
    def store_embeddings(self, document_id: int, embeddings: np.ndarray) -> None:
        """Store embeddings for document chunks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get chunk IDs for this document
            cursor.execute(
                "SELECT id FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,)
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]
            
            # Clear existing embeddings
            cursor.execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
                (document_id,)
            )
            
            # Store embeddings
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                embedding_blob = pickle.dumps(embedding.astype('float32'))
                cursor.execute('''
                    INSERT INTO embeddings (chunk_id, embedding_vector)
                    VALUES (?, ?)
                ''', (chunk_id, embedding_blob))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_embeddings(self, document_id: int) -> Optional[np.ndarray]:
        """Get embeddings for document."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT e.embedding_vector
            FROM embeddings e
            JOIN chunks c ON e.chunk_id = c.id
            WHERE c.document_id = ?
            ORDER BY c.chunk_index
        ''', (document_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return None
        
        # Deserialize embeddings
        embeddings = []
        for row in results:
            embedding = pickle.loads(row[0])
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def store_faiss_index(self, document_id: int, index_data: bytes, dimension: int) -> None:
        """Store FAISS index for document."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO faiss_indices (document_id, index_data, dimension)
            VALUES (?, ?, ?)
        ''', (document_id, index_data, dimension))
        
        conn.commit()
        conn.close()
    
    def get_faiss_index(self, document_id: int) -> Optional[bytes]:
        """Get FAISS index for document."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT index_data FROM faiss_indices WHERE document_id = ?",
            (document_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_chunks_by_document_id(self, document_id: int) -> List[str]:
        """Get all chunks for a document."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,)
        )
        chunks = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return chunks
    
    def migrate_from_files(self, filename: str) -> bool:
        """Migrate existing file-based data to database."""
        base_name = filename[:-4] if filename.lower().endswith(".pdf") else filename
        txt_file = base_name + ".txt" if filename.lower().endswith(".pdf") else filename
        embeddings_file = txt_file + ".embeddings.npy"
        faiss_file = txt_file + ".faiss"
        
        # Check if files exist
        if not os.path.exists(txt_file):
            return False
        
        # Read text content
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split into chunks (using same logic as original)
        def split_into_chunks(text, max_words=200):
            words = text.split()
            chunks = []
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk)
            return chunks
        
        chunks = split_into_chunks(content)
        
        # Store document and chunks
        document_id = self.store_document(filename, content, chunks)
        
        # Migrate embeddings if they exist
        if os.path.exists(embeddings_file):
            embeddings = np.load(embeddings_file)
            self.store_embeddings(document_id, embeddings)
        
        # Migrate FAISS index if it exists
        if os.path.exists(faiss_file):
            with open(faiss_file, "rb") as f:
                index_data = f.read()
            dimension = embeddings.shape[1] if 'embeddings' in locals() else 384  # default dimension
            self.store_faiss_index(document_id, index_data, dimension)
        
        return True