# Vector Database Loader for AI Agents

A Python system for reading input files and storing data in a vector database that can be used by AI agents for retrieval-augmented generation (RAG) applications.

## Features

- **Multi-format Support**: Handles text files (.txt), CSV files (.csv), and JSON files (.json)
- **Intelligent Chunking**: Automatically splits large texts into overlapping chunks for better retrieval
- **Vector Embeddings**: Uses SentenceTransformers to create semantic embeddings
- **Persistent Storage**: Uses ChromaDB for efficient vector storage and retrieval
- **AI Agent Integration**: Includes example AI agent implementation
- **Metadata Tracking**: Preserves source information and file metadata

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from vector_db_loader import VectorDBLoader

# Initialize the vector database loader
loader = VectorDBLoader(db_path="./my_vector_db", collection_name="documents")

# Load a file into the vector database
count = loader.load_file("my_document.txt")
print(f"Loaded {count} documents")

# Search the database
results = loader.search("artificial intelligence", n_results=5)
```

### AI Agent Integration

```python
from vector_db_loader import VectorDBLoader
from example_usage import AIAgent

# Initialize vector database and AI agent
loader = VectorDBLoader()
agent = AIAgent(loader)

# Load your knowledge base
loader.load_file("knowledge_base.txt")

# Query the AI agent
response = agent.query("What is machine learning?")
print(response)
```

## File Format Support

### Text Files (.txt)
- Automatically chunks long texts into overlapping segments
- Preserves document structure and context
- Ideal for documentation, articles, and books

### CSV Files (.csv)
- Converts each row into a searchable document
- Preserves column information as metadata
- Perfect for structured data like customer records, product catalogs

### JSON Files (.json)
- Handles both JSON objects and arrays
- Converts nested structures into searchable text
- Maintains original data structure in metadata

## API Reference

### VectorDBLoader Class

#### `__init__(db_path="./vector_db", collection_name="documents")`
Initialize the vector database loader.

#### `load_file(file_path) -> int`
Load a file into the vector database. Returns the number of documents added.

#### `search(query, n_results=5) -> Dict`
Search the vector database for relevant documents.

#### `add_documents(documents) -> int`
Add a list of documents to the vector database.

#### `get_collection_info() -> Dict`
Get information about the current collection.

### AIAgent Class

#### `__init__(vector_db_loader)`
Initialize the AI agent with a vector database loader.

#### `query(question, context_limit=3) -> str`
Query the AI agent using the vector database for context.

## Examples

### Example 1: Loading Multiple Files

```python
from vector_db_loader import VectorDBLoader

loader = VectorDBLoader()

files = ["document1.txt", "data.csv", "config.json"]
total_docs = 0

for file_path in files:
    count = loader.load_file(file_path)
    total_docs += count
    print(f"Loaded {count} documents from {file_path}")

print(f"Total documents: {total_docs}")
```

### Example 2: Custom Document Processing

```python
from vector_db_loader import VectorDBLoader

loader = VectorDBLoader()

# Custom documents
documents = [
    {
        'content': 'Python is a programming language.',
        'metadata': {'topic': 'programming', 'difficulty': 'beginner'}
    },
    {
        'content': 'Machine learning uses algorithms to find patterns.',
        'metadata': {'topic': 'AI', 'difficulty': 'intermediate'}
    }
]

count = loader.add_documents(documents)
print(f"Added {count} custom documents")
```

### Example 3: Advanced Search

```python
from vector_db_loader import VectorDBLoader

loader = VectorDBLoader()
loader.load_file("knowledge_base.txt")

# Search with different parameters
results = loader.search("neural networks", n_results=3)

for i, (doc, metadata, distance) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0], 
    results['distances'][0]
)):
    similarity = 1 - distance
    print(f"Result {i+1} (Similarity: {similarity:.3f}):")
    print(f"Source: {metadata['source']}")
    print(f"Content: {doc[:100]}...")
    print()
```

## Running the Examples

### Basic Demo
```bash
python vector_db_loader.py
```

This will:
- Create sample files (text, CSV, JSON)
- Load them into the vector database
- Perform example searches
- Display results with similarity scores

### AI Agent Demo
```bash
python example_usage.py
```

This will:
- Set up a knowledge base with AI-related content
- Create an AI agent that uses the vector database
- Demonstrate query processing with context retrieval

## Configuration

### Vector Database Settings
- **Database Path**: Location where ChromaDB stores the vector database
- **Collection Name**: Name of the collection within the database
- **Embedding Model**: Default is 'all-MiniLM-L6-v2' (can be changed)

### Text Chunking Settings
- **Chunk Size**: Default 1000 characters
- **Overlap**: Default 100 characters overlap between chunks
- **Sentence Boundary**: Attempts to break at sentence endings

## Best Practices

1. **File Organization**: Keep related documents in the same collection
2. **Chunking**: Adjust chunk size based on your content type
3. **Metadata**: Use descriptive metadata for better filtering
4. **Regular Updates**: Rebuild the database when source files change
5. **Performance**: Consider using GPU acceleration for large datasets

## Troubleshooting

### Common Issues

1. **Import Errors**: Install dependencies with `pip install -r requirements.txt`
2. **Memory Issues**: Reduce chunk size or process files in batches
3. **Search Quality**: Try different embedding models or adjust chunk sizes
4. **Database Errors**: Check database path permissions and disk space

### Performance Tips

- Use SSD storage for better I/O performance
- Consider using a GPU for faster embedding generation
- Batch process multiple files when possible
- Regular database maintenance and cleanup

## Advanced Usage

### Custom Embedding Models

```python
from sentence_transformers import SentenceTransformer

# Use a different embedding model
loader = VectorDBLoader()
loader.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### Database Migration

```python
# Export collection
results = loader.collection.get()

# Import to new collection
new_loader = VectorDBLoader(collection_name="new_collection")
# Process and add documents...
```

## Dependencies

- `chromadb>=0.4.0`: Vector database
- `sentence-transformers>=2.2.0`: Text embeddings
- `numpy>=1.21.0`: Numerical computations
- `pandas>=1.3.0`: Data manipulation
- `torch>=1.9.0`: PyTorch for neural networks
- `transformers>=4.20.0`: Transformer models

## License

This project is open source and available under the MIT License.