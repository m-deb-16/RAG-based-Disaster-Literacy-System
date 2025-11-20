"""
Vector Store Module for Disaster Literacy RAG System
Implements FAISS-based vector store with sentence-transformers embeddings
References: Lines 37-40, 100-101, 246 (FAISS + sentence-transformers for offline)
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    VECTOR_INDEX_PATH,
    CHUNK_METADATA_PATH,
    VECTOR_STORE_DIR
)
from error_handler import error_handler, RetrievalError


class VectorStore:
    """
    FAISS-based vector store for chunk embeddings
    References: Lines 37-40 (Embeddings index with FAISS/HNSWLIB)
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize vector store with embedding model
        References: Line 100, 289 (all-MiniLM-L6-v2 for fast, low RAM)
        """
        self.model_name = model_name
        self.embedding_dim = EMBEDDING_DIMENSION
        self.index_path = Path(VECTOR_INDEX_PATH)
        self.metadata_path = Path(CHUNK_METADATA_PATH)
        
        # Ensure directory exists
        Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
        
        # Load embedding model - References: Line 100
        error_handler.logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Initialize or load FAISS index
        self.index = None
        self.chunk_metadata = []
        self._load_or_create_index()
        
    def _load_or_create_index(self) -> None:
        """
        Load existing index or create new one
        References: Lines 37-39 (Local vector index storage)
        """
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                    
                error_handler.logger.info(
                    f"Loaded existing index with {len(self.chunk_metadata)} chunks"
                )
            except Exception as e:
                error_handler.logger.warning(f"Failed to load index: {e}. Creating new.")
                self._create_new_index()
        else:
            self._create_new_index()
            
    def _create_new_index(self) -> None:
        """
        Create new FAISS index - References: Line 101 (FAISS flat for small KB)
        """
        # Using Flat index for exact search (good for small-medium KB)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunk_metadata = []
        error_handler.logger.info("Created new FAISS Flat index")
        
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to vector store with embeddings
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata
        """
        if not chunks:
            return
            
        try:
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings - References: Line 100
            error_handler.logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.encoder.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Normalize embeddings for cosine similarity (optional but recommended)
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata - References: Line 39 (mapping chunk_id → text + metadata)
            self.chunk_metadata.extend(chunks)
            
            # Save index and metadata
            self._save_index()
            
            error_handler.logger.info(
                f"Added {len(chunks)} chunks to vector store. "
                f"Total: {len(self.chunk_metadata)} chunks"
            )
            
        except Exception as e:
            error_handler.logger.error(f"Failed to add chunks: {e}")
            raise RetrievalError(f"Failed to add chunks to vector store: {e}")
            
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k similar chunks
        References: Lines 42-48 (Dense retrieval returning top-k chunks)
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., disaster_type)
            
        Returns:
            List of top-k chunks with scores
        """
        if self.index.ntotal == 0:
            error_handler.logger.warning("Vector store is empty")
            raise RetrievalError(
                "No documents in knowledge base. "
                "Please upload documents via Admin Panel and reinitialize the system."
            )
            
        try:
            # Encode query
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True
            )
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search index
            # Get more results for filtering if needed
            search_k = top_k * 3 if filter_metadata else top_k
            distances, indices = self.index.search(
                query_embedding.astype('float32'),
                min(search_k, self.index.ntotal)
            )
            
            # Retrieve chunks with metadata
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunk_metadata):
                    chunk = self.chunk_metadata[idx].copy()
                    
                    # Convert L2 distance to similarity score (0-1 range)
                    # Since vectors are normalized, L2 dist relates to cosine similarity
                    similarity = 1 / (1 + dist)
                    chunk['score'] = float(similarity)
                    
                    # Apply metadata filters if specified
                    if filter_metadata:
                        match = all(
                            chunk.get(key) == value
                            for key, value in filter_metadata.items()
                        )
                        if not match:
                            continue
                            
                    results.append(chunk)
                    
                    if len(results) >= top_k:
                        break
                        
            if not results:
                error_handler.logger.warning(f"No results found for query: {query[:50]}...")
                
            return results
            
        except Exception as e:
            error_handler.logger.error(f"Search failed: {e}")
            raise RetrievalError(f"Failed to search vector store: {e}")
            
    def _save_index(self) -> None:
        """
        Save FAISS index and metadata to disk
        References: Line 39 (Store mapping and index)
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
                
            error_handler.logger.info("Vector store saved successfully")
            
        except Exception as e:
            error_handler.logger.error(f"Failed to save index: {e}")
            raise
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        """
        return {
            "total_chunks": len(self.chunk_metadata),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "index_size_mb": self.index_path.stat().st_size / (1024 * 1024)
                if self.index_path.exists() else 0
        }
        
    def clear(self) -> None:
        """
        Clear vector store (for testing or re-indexing)
        """
        self._create_new_index()
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        error_handler.logger.info("Vector store cleared")


    def delete_document(self, source_name: str) -> bool:
        """
        Delete all chunks associated with a specific document source
        
        Args:
            source_name: Name of the source file to delete
            
        Returns:
            bool: True if successful
        """
        try:
            # Filter out chunks belonging to the document
            initial_count = len(self.chunk_metadata)
            new_metadata = [
                chunk for chunk in self.chunk_metadata 
                if chunk.get("source") != source_name
            ]
            
            if len(new_metadata) == initial_count:
                error_handler.logger.warning(f"No chunks found for document: {source_name}")
                return False
                
            # Rebuild index with remaining chunks
            # This is necessary because removing from Flat index by ID is complex
            # and we need to keep metadata in sync
            
            # Reset index and metadata
            self._create_new_index()
            
            # Add remaining chunks back
            if new_metadata:
                # Extract texts for embedding
                texts = [chunk['text'] for chunk in new_metadata]
                
                # Generate embeddings
                error_handler.logger.info(f"Re-indexing {len(texts)} chunks after deletion")
                embeddings = self.encoder.encode(
                    texts,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                
                # Normalize embeddings
                faiss.normalize_L2(embeddings)
                
                # Add to FAISS index
                self.index.add(embeddings.astype('float32'))
                
                # Store metadata
                self.chunk_metadata = new_metadata
                
                # Save index and metadata
                self._save_index()
            else:
                # If no chunks left, just save empty state
                self._save_index()
                
            error_handler.logger.info(
                f"Deleted document '{source_name}' from vector store. "
                f"Removed {initial_count - len(new_metadata)} chunks."
            )
            return True
            
        except Exception as e:
            error_handler.logger.error(f"Failed to delete document {source_name}: {e}")
            raise RetrievalError(f"Failed to delete document from vector store: {e}")

