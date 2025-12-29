from ai_zat.rag import rag_manager

def test_rag_ingestion():
    """Test that text can be ingested and retrieved."""
    # Clear first
    rag_manager.clear_database()
    
    sample_text = "The quick brown fox jumps over the lazy dog."
    rag_manager.ingest_text(sample_text)
    
    results = rag_manager.retrieve("brown fox")
    assert len(results) > 0
    assert "fox" in results[0].page_content
