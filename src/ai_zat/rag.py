import json
import logging
from pathlib import Path
from typing import Optional

from flashrank import Ranker, RerankRequest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIRECTORY = Path(".chroma_db")
CORPUS_PATH = Path(".corpus_metadata.json")
COLLECTION_NAME = "journal_collection"

class RAGManager:
    """Manages the Vector Database, Retrieval, and Re-ranking operations."""
    
    def __init__(self) -> None:
        """Initialize RAG Manager with local embeddings and re-ranker."""
        # Use a lightweight local model for speed/privacy
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
        
        # Initialize FlashRank Re-ranker
        try:
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".flashrank_cache")
            logger.info("FlashRank Re-ranker initialized.")
        except Exception as e:
            logger.warning(f"FlashRank not available: {e}")
            self.reranker = None

        # Hybrid Search: BM25
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_docs: list[Document] = []
        self._load_corpus()

    def _load_corpus(self) -> None:
        """Load corpus from JSON for BM25 persistence."""
        if CORPUS_PATH.exists():
            try:
                with open(CORPUS_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                    self.corpus_docs = [Document(**d) for d in data]
                self._build_bm25()
                logger.info(f"Loaded {len(self.corpus_docs)} documents into BM25.")
            except Exception as e:
                logger.error(f"Failed to load corpus: {e}")

    def _save_corpus(self) -> None:
        """Save corpus to JSON."""
        try:
            with open(CORPUS_PATH, "w", encoding="utf-8") as f:
                data = [doc.dict() for doc in self.corpus_docs]
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save corpus: {e}")

    def _build_bm25(self) -> None:
        """Build BM25 index from corpus."""
        if not self.corpus_docs:
            return
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.corpus_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def ingest_documents(self, documents: list[Document]) -> None:
        """
        Ingest a list of Documents with metadata into the vector store.
        
        Args:
            documents: List of LangChain Documents.
        """
        if not documents:
            logger.warning("No documents provided for ingestion.")
            return

        try:
            # Split
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} chunks from {len(documents)} pages.")
            
            # Index Chroma
            self.vector_store.add_documents(documents=splits)
            
            # Update BM25
            self.corpus_docs.extend(splits)
            self._save_corpus()
            self._build_bm25()
            
            logger.info("Successfully added chunks to Hybrid Store.")
            
        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            raise

    def retrieve(self, query: str, k: int = 4, rerank: bool = True) -> list[Document]:
        """
        Retrieve relevant contexts using Hybrid Search (Vector + BM25) and RRF.
        """
        try:
            # 1. Vector Search
            vector_results = self.vector_store.similarity_search(query, k=k * 2)
            
            # 2. BM25 Search
            bm25_results: list[Document] = []
            if self.bm25:
                tokenized_query = query.lower().split()
                bm25_results = self.bm25.get_top_n(tokenized_query, self.corpus_docs, n=k * 2)

            # 3. Reciprocal Rank Fusion (RRF)
            candidates = self._reciprocal_rank_fusion(vector_results, bm25_results, k=k * 3)
            logger.info(f"Hybrid retrieval found {len(candidates)} RRF candidates.")

            # 4. Re-rank if enabled
            if self.reranker and rerank and candidates:
                passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(candidates)]
                rerank_request = RerankRequest(query=query, passages=passages)
                reranked = self.reranker.rerank(rerank_request)
                
                reranked_docs = [candidates[int(r["id"])] for r in reranked[:k]]
                logger.info(f"Re-ranked to top {len(reranked_docs)} results.")
                return reranked_docs
            
            return candidates[:k]
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _reciprocal_rank_fusion(self, vector_docs: list[Document], bm25_docs: list[Document], k: int = 10, c: int = 60) -> list[Document]:
        """Combine results using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = doc.page_content # Use content as ID for simplicity
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + c)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.page_content
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + c)
            doc_map[doc_id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids[:k]]

    def clear_database(self) -> None:
        """Clear vector store and BM25 corpus."""
        try:
            self.vector_store.delete_collection()
            self.corpus_docs = []
            if CORPUS_PATH.exists():
                CORPUS_PATH.unlink()
            self.bm25 = None
            logger.info("Hybrid database cleared.")
            # Re-init Chroma
            self.vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                persist_directory=str(PERSIST_DIRECTORY)
            )
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            

# Global instance
rag_manager = RAGManager()
