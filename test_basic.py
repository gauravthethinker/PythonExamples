#!/usr/bin/env python3
"""
Basic test script to verify the vector database loader structure
This script tests the basic functionality without requiring heavy dependencies.
"""

import json
import csv
import hashlib
from pathlib import Path
from typing import List, Dict, Any


class MockVectorDBLoader:
    """Mock version of VectorDBLoader for testing without dependencies."""
    
    def __init__(self, db_path: str = "./test_vector_db", collection_name: str = "test_documents"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.documents = []  # Store documents in memory for testing
        print(f"Mock VectorDB initialized: {collection_name} at {db_path}")
    
    def _generate_id(self, text: str, metadata: Dict = None) -> str:
        """Generate a unique ID for a document."""
        content = text + str(metadata or {})
        return hashlib.md5(content.encode()).hexdigest()
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for better retrieval."""
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
                content = "\n".join([f"{key}: {value}" for key, value in row.items() if value])
                
                doc = {
                    'content': content,
                    'metadata': {
                        'source': file_path,
                        'file_type': 'csv',
                        'row_index': row_index,
                        **{k: v for k, v in row.items() if v}
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
        """Load a file into the mock database."""
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
            documents = self.read_text_file(str(file_path))
        
        # Add to mock database
        self.documents.extend(documents)
        print(f"Successfully added {len(documents)} documents to mock database.")
        return len(documents)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the mock collection."""
        return {
            'collection_name': self.collection_name,
            'document_count': len(self.documents),
            'db_path': self.db_path
        }
    
    def search_mock(self, query: str) -> List[Dict[str, Any]]:
        """Simple text-based search for testing."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            if query_lower in doc['content'].lower():
                results.append(doc)
        
        return results[:5]  # Return top 5 matches


def create_test_files():
    """Create test files for demonstration."""
    
    # Test text file
    text_content = """
    Artificial Intelligence (AI) is revolutionizing how we work and live.
    Machine learning algorithms can identify patterns in large datasets.
    Natural language processing enables computers to understand human speech.
    Computer vision allows machines to interpret visual information.
    Deep learning uses neural networks to solve complex problems.
    """
    
    with open('test_document.txt', 'w') as f:
        f.write(text_content)
    
    # Test CSV file
    csv_data = [
        {'name': 'Alice Smith', 'role': 'Data Scientist', 'department': 'AI Research'},
        {'name': 'Bob Johnson', 'role': 'ML Engineer', 'department': 'Engineering'},
        {'name': 'Carol Davis', 'role': 'Product Manager', 'department': 'Product'}
    ]
    
    with open('test_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'role', 'department'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    # Test JSON file
    json_data = {
        'company': 'AI Solutions Inc.',
        'employees': [
            {'id': 1, 'name': 'John Doe', 'skills': ['Python', 'Machine Learning']},
            {'id': 2, 'name': 'Jane Smith', 'skills': ['Data Science', 'Statistics']}
        ],
        'projects': [
            {'name': 'AI Chatbot', 'status': 'active', 'budget': 50000},
            {'name': 'Recommendation System', 'status': 'completed', 'budget': 75000}
        ]
    }
    
    with open('test_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)


def test_vector_db_loader():
    """Test the mock vector database loader."""
    
    print("🧪 TESTING VECTOR DATABASE LOADER")
    print("=" * 50)
    
    # Create test files
    print("📁 Creating test files...")
    create_test_files()
    
    # Initialize mock loader
    loader = MockVectorDBLoader()
    
    # Test loading different file types
    test_files = ['test_document.txt', 'test_data.csv', 'test_data.json']
    total_docs = 0
    
    print("\n📚 Loading test files...")
    for file_path in test_files:
        try:
            count = loader.load_file(file_path)
            total_docs += count
            print(f"✅ Loaded {count} documents from {file_path}")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    # Display collection info
    print(f"\n📊 Collection Information:")
    info = loader.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test search functionality
    print(f"\n🔍 Testing search functionality...")
    search_queries = [
        "artificial intelligence",
        "machine learning", 
        "data scientist",
        "Python"
    ]
    
    for query in search_queries:
        results = loader.search_mock(query)
        print(f"\nQuery: '{query}' -> Found {len(results)} results")
        for i, result in enumerate(results[:2]):  # Show first 2 results
            print(f"  Result {i+1}: {result['content'][:100]}...")
    
    # Test chunking functionality
    print(f"\n🔧 Testing text chunking...")
    long_text = "This is a test. " * 100  # Create a long text
    chunks = loader._chunk_text(long_text, chunk_size=50, overlap=10)
    print(f"  Original text length: {len(long_text)} characters")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Average chunk length: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f} characters")
    
    # Cleanup test files
    print(f"\n🧹 Cleaning up test files...")
    for file_path in test_files:
        try:
            Path(file_path).unlink()
            print(f"  ✅ Removed {file_path}")
        except Exception as e:
            print(f"  ❌ Error removing {file_path}: {e}")
    
    print(f"\n✅ All tests completed successfully!")
    print(f"\nTo run the full version with embeddings:")
    print(f"1. Run: ./setup.sh (or sudo ./setup.sh if needed)")
    print(f"2. Activate environment: source vector_db_env/bin/activate") 
    print(f"3. Run: python vector_db_loader.py")


if __name__ == "__main__":
    test_vector_db_loader()