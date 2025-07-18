#!/usr/bin/env python3
"""
Vector Database Loader
A Python script to read input files and store data in a vector database for AI agent use.
Supports text, CSV, and JSON file formats.
"""

import os
import json
import csv
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed. Installing...")
    os.system("pip install chromadb")
    import chromadb
    from chromadb.config import Settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformers not installed. Installing...")
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer


class VectorDBLoader:
    """Class to handle loading data into a vector database."""
    
    def __init__(self, db_path: str = "./vector_db", collection_name: str = "documents"):
        """
        Initialize the VectorDBLoader.
        
        Args:
            db_path: Path to store the ChromaDB database
            collection_name: Name of the collection in the database
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded successfully!")
    
    def _generate_id(self, text: str, metadata: Dict = None) -> str:
        """Generate a unique ID for a document."""
        content = text + str(metadata or {})
        return hashlib.md5(content.encode()).hexdigest()
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def read_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Read a plain text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        chunks = self._chunk_text(content)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = {
                'content': chunk,
                'metadata': {
                    'source': file_path,
                    'file_type': 'text',
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            }
            documents.append(doc)
        
        return documents
    
    def read_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Read a CSV file and convert rows to documents."""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_index, row in enumerate(reader):
                # Convert row to text representation
                content = "\n".join([f"{key}: {value}" for key, value in row.items() if value])
                
                doc = {
                    'content': content,
                    'metadata': {
                        'source': file_path,
                        'file_type': 'csv',
                        'row_index': row_index,
                        **{k: v for k, v in row.items() if v}  # Include non-empty values as metadata
                    }
                }
                documents.append(doc)
        
        return documents
    
    def read_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Read a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        documents = []
        
        if isinstance(data, list):
            # Handle JSON array
            for i, item in enumerate(data):
                content = json.dumps(item, indent=2)
                doc = {
                    'content': content,
                    'metadata': {
                        'source': file_path,
                        'file_type': 'json',
                        'item_index': i,
                        'total_items': len(data)
                    }
                }
                documents.append(doc)
        else:
            # Handle single JSON object
            content = json.dumps(data, indent=2)
            chunks = self._chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                doc = {
                    'content': chunk,
                    'metadata': {
                        'source': file_path,
                        'file_type': 'json',
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                }
                documents.append(doc)
        
        return documents
    
    def load_file(self, file_path: str) -> int:
        """
        Load a file into the vector database.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Number of documents added
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading file: {file_path}")
        
        # Determine file type and read accordingly
        if file_path.suffix.lower() == '.csv':
            documents = self.read_csv_file(str(file_path))
        elif file_path.suffix.lower() == '.json':
            documents = self.read_json_file(str(file_path))
        else:
            # Treat as text file
            documents = self.read_text_file(str(file_path))
        
        # Add documents to vector database
        return self.add_documents(documents)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add documents to the vector database."""
        if not documents:
            return 0
        
        print(f"Processing {len(documents)} documents...")
        
        # Prepare data for ChromaDB
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [self._generate_id(doc['content'], doc['metadata']) for doc in documents]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully added {len(documents)} documents to the vector database.")
        return len(documents)
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search the vector database.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        print(f"Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'db_path': self.db_path
        }


def main():
    """Example usage of the VectorDBLoader."""
    
    # Create sample data files for demonstration
    sample_text = """
    Artificial Intelligence (AI) is a transformative technology that simulates human intelligence in machines.
    Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.
    Natural Language Processing (NLP) allows machines to understand, interpret, and generate human language.
    Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.
    """
    
    sample_csv_data = [
        {'name': 'Alice', 'age': '30', 'city': 'New York', 'occupation': 'Engineer'},
        {'name': 'Bob', 'age': '25', 'city': 'San Francisco', 'occupation': 'Designer'},
        {'name': 'Charlie', 'age': '35', 'city': 'Chicago', 'occupation': 'Manager'}
    ]
    
    sample_json_data = {
        'products': [
            {'id': 1, 'name': 'Laptop', 'price': 999.99, 'category': 'Electronics'},
            {'id': 2, 'name': 'Book', 'price': 19.99, 'category': 'Education'},
            {'id': 3, 'name': 'Phone', 'price': 699.99, 'category': 'Electronics'}
        ]
    }
    
    # Create sample files
    with open('sample_text.txt', 'w') as f:
        f.write(sample_text)
    
    with open('sample_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'age', 'city', 'occupation'])
        writer.writeheader()
        writer.writerows(sample_csv_data)
    
    with open('sample_data.json', 'w') as f:
        json.dump(sample_json_data, f, indent=2)
    
    # Initialize the vector database loader
    loader = VectorDBLoader()
    
    # Load sample files
    print("\n" + "="*50)
    print("LOADING FILES INTO VECTOR DATABASE")
    print("="*50)
    
    total_docs = 0
    for file_path in ['sample_text.txt', 'sample_data.csv', 'sample_data.json']:
        try:
            count = loader.load_file(file_path)
            total_docs += count
            print(f"✓ Loaded {count} documents from {file_path}")
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
    
    # Display collection info
    print("\n" + "="*50)
    print("DATABASE INFORMATION")
    print("="*50)
    info = loader.get_collection_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Example searches
    print("\n" + "="*50)
    print("EXAMPLE SEARCHES")
    print("="*50)
    
    search_queries = [
        "artificial intelligence and machine learning",
        "people from San Francisco", 
        "electronic products"
    ]
    
    for query in search_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        results = loader.search(query, n_results=3)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"Result {i+1} (similarity: {1-distance:.3f}):")
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Content: {doc[:100]}...")
            print()


if __name__ == "__main__":
    main()