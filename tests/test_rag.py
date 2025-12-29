from langchain_core.documents import Document
from ai_zat.rag import rag_manager

def test_hybrid_rag_ingestion():
    """Test that text can be ingested and retrieved via Hybrid Search."""
    # Clear first
    rag_manager.clear_database()
    
    sample_docs = [
        Document(
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={"source": "test", "page": 1}
        )
    ]
    rag_manager.ingest_documents(sample_docs)
    
    # Test Hybrid Retrieval
    results = rag_manager.retrieve("brown fox")
    assert len(results) > 0
    assert "fox" in results[0].page_content
    assert results[0].metadata["page"] == 1
