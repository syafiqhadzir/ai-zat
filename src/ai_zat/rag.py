"""RAG module for handling document chunking, embedding, and retrieval."""
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIRECTORY = Path(".chroma_db")
COLLECTION_NAME = "journal_collection"

class RAGManager:
    """Manages the Vector Database and Retrieval operations."""
    
    def __init__(self) -> None:
        """Initialize RAG Manager with local embeddings."""
        # Use a lightweight local mode for speed/privacy
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize Vector Store
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            persist_directory=str(PERSIST_DIRECTORY)
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

    def ingest_text(self, text: str, source: str = "journal") -> None:
        """
        Ingest raw text into the vector store.
        
        Args:
            text: Full text content.
            source: Source metadata identifier.
        """
        if not text:
            logger.warning("No text provided for ingestion.")
            return

        try:
            # Create Documents
            docs = [Document(page_content=text, metadata={"source": source})]
            
            # Split
            splits = self.text_splitter.split_documents(docs)
            logger.info(f"Created {len(splits)} chunks from text.")
            
            # Index
            self.vector_store.add_documents(documents=splits)
            logger.info("Successfully added chunks to ChromaDB.")
            
        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            raise

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant contexts for a query.
        
        Args:
            query: User question.
            k: Number of chunks to retrieve.
            
        Returns:
            List of relevant Documents.
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
            
    def clear_database(self) -> None:
        """Clear the existing vector store."""
        try:
            self.vector_store.delete_collection()
            logger.info("Vector database cleared.")
            # Re-init after delete
            self.vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                persist_directory=str(PERSIST_DIRECTORY)
            )
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")

# Global instance
rag_manager = RAGManager()
