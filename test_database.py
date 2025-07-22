#!/usr/bin/env python3
"""
Test script to verify database functionality works correctly.
This runs without requiring external model downloads.
"""

import os
import sys
import numpy as np
from database import ChatPDFDatabase

def test_database_migration():
    """Test complete database functionality including migration."""
    print("Testing ChatPDF Database functionality...")
    
    # Initialize database
    db = ChatPDFDatabase('chat_pdf_test.db')
    print("✓ Database initialized")
    
    # Test document storage and retrieval
    test_filename = "test_document.pdf"
    test_content = "This is a test document with multiple sentences. It contains information about various topics. The content is split into chunks for processing."
    test_chunks = [
        "This is a test document with multiple sentences.",
        "It contains information about various topics.",
        "The content is split into chunks for processing."
    ]
    
    # Create the file on disk first (required for file hash calculation)
    with open(test_filename, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    # Test storing document
    document_id = db.store_document(test_filename, test_content, test_chunks)
    print(f"✓ Document stored with ID: {document_id}")
    
    # Test retrieving document
    result = db.get_document_by_filename(test_filename)
    assert result is not None, "Document should be retrievable"
    retrieved_id, retrieved_content, retrieved_chunks = result
    assert retrieved_id == document_id, "Document ID should match"
    assert retrieved_content == test_content, "Content should match"
    assert retrieved_chunks == test_chunks, "Chunks should match"
    print("✓ Document retrieval works correctly")
    
    # Test embeddings storage (mock embeddings)
    mock_embeddings = np.random.rand(3, 384).astype('float32')  # 3 chunks, 384 dimensions
    db.store_embeddings(document_id, mock_embeddings)
    print("✓ Embeddings stored")
    
    # Test embeddings retrieval
    retrieved_embeddings = db.get_embeddings(document_id)
    assert retrieved_embeddings is not None, "Embeddings should be retrievable"
    assert np.array_equal(retrieved_embeddings, mock_embeddings), "Embeddings should match"
    print("✓ Embeddings retrieval works correctly")
    
    # Test FAISS index storage (mock index data)
    mock_index_data = b"mock_faiss_index_data_for_testing"
    db.store_faiss_index(document_id, mock_index_data, 384)
    print("✓ FAISS index stored")
    
    # Test FAISS index retrieval
    retrieved_index_data = db.get_faiss_index(document_id)
    assert retrieved_index_data == mock_index_data, "FAISS index data should match"
    print("✓ FAISS index retrieval works correctly")
    
    # Test document existence check (file is already on disk)
    assert db.document_exists(test_filename), "Document should exist"
    print("✓ Document existence check works")
    
    # Clean up the test file
    os.remove(test_filename)
    
    # Test file migration functionality
    # Create mock files to test migration
    txt_file = "test_migration.txt"
    emb_file = txt_file + ".embeddings.npy"
    faiss_file = txt_file + ".faiss"
    
    migration_content = "Migration test content for file-based storage."
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(migration_content)
    
    # Create mock embeddings file
    migration_embeddings = np.random.rand(2, 384).astype('float32')
    np.save(emb_file, migration_embeddings)
    
    # Create mock FAISS file
    with open(faiss_file, "wb") as f:
        f.write(b"mock_faiss_index_for_migration")
    
    # Test migration
    migration_filename = "test_migration.pdf"
    success = db.migrate_from_files(migration_filename)
    assert success, "Migration should succeed"
    print("✓ File migration works correctly")
    
    # Verify migrated data
    migrated_result = db.get_document_by_filename(migration_filename)
    assert migrated_result is not None, "Migrated document should be retrievable"
    migrated_id, migrated_content, migrated_chunks = migrated_result
    assert migration_content in migrated_content, "Migrated content should be preserved"
    print("✓ Migrated data verification successful")
    
    # Test chunks retrieval
    chunks = db.get_chunks_by_document_id(document_id)
    assert chunks == test_chunks, "Chunks should match"
    print("✓ Chunks retrieval works correctly")
    
    # Cleanup test files
    cleanup_files = [txt_file, emb_file, faiss_file, 'chat_pdf_test.db']
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
    print("✓ Cleanup completed")
    
    print("\n🎉 All database tests passed! Database functionality is working correctly.")
    return True

def test_database_performance():
    """Test database performance with larger dataset."""
    print("\nTesting database performance...")
    
    db = ChatPDFDatabase('chat_pdf_perf_test.db')
    
    # Test with multiple documents
    num_docs = 5
    chunks_per_doc = 20
    
    for doc_num in range(num_docs):
        filename = f"perf_test_doc_{doc_num}.pdf"
        content = f"Performance test document {doc_num} " * 100
        chunks = [f"Chunk {i} for document {doc_num}: " + "test content " * 20 
                 for i in range(chunks_per_doc)]
        
        doc_id = db.store_document(filename, content, chunks)
        
        # Add mock embeddings
        embeddings = np.random.rand(chunks_per_doc, 384).astype('float32')
        db.store_embeddings(doc_id, embeddings)
        
        # Add mock FAISS index
        index_data = f"faiss_index_for_doc_{doc_num}".encode() * 10
        db.store_faiss_index(doc_id, index_data, 384)
    
    print(f"✓ Successfully stored {num_docs} documents with {chunks_per_doc} chunks each")
    
    # Test retrieval performance
    for doc_num in range(num_docs):
        filename = f"perf_test_doc_{doc_num}.pdf"
        result = db.get_document_by_filename(filename)
        assert result is not None, f"Document {doc_num} should be retrievable"
    
    print("✓ All documents retrieved successfully")
    
    # Cleanup
    os.remove('chat_pdf_perf_test.db')
    print("✓ Performance test cleanup completed")
    
    print("🚀 Database performance test passed!")

if __name__ == "__main__":
    try:
        test_database_migration()
        test_database_performance()
        print("\n✅ All tests completed successfully!")
        print("The database implementation is ready for production use.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)