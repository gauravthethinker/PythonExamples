#!/usr/bin/env python3
"""
Example Usage of Vector Database Loader for AI Agents
This script demonstrates how to use the VectorDBLoader for AI agent applications.
"""

from vector_db_loader import VectorDBLoader
import json


class AIAgent:
    """Simple AI Agent that uses vector database for retrieval."""
    
    def __init__(self, vector_db_loader: VectorDBLoader):
        self.vector_db = vector_db_loader
    
    def query(self, question: str, context_limit: int = 3) -> str:
        """
        Answer a question using the vector database as context.
        
        Args:
            question: User's question
            context_limit: Number of relevant documents to retrieve
            
        Returns:
            Formatted response with context
        """
        # Search for relevant documents
        results = self.vector_db.search(question, n_results=context_limit)
        
        if not results['documents'][0]:
            return "No relevant information found in the database."
        
        # Format the response
        response = f"Question: {question}\n\n"
        response += "Relevant Information:\n"
        response += "=" * 50 + "\n"
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity = 1 - distance
            response += f"\nSource {i+1} (Similarity: {similarity:.3f}):\n"
            response += f"File: {metadata.get('source', 'Unknown')}\n"
            response += f"Content: {doc}\n"
            response += "-" * 30 + "\n"
        
        return response


def demonstrate_ai_agent():
    """Demonstrate the AI agent functionality."""
    
    print("🤖 AI AGENT WITH VECTOR DATABASE DEMO")
    print("=" * 50)
    
    # Initialize vector database loader
    loader = VectorDBLoader(db_path="./ai_agent_db", collection_name="knowledge_base")
    
    # Create sample knowledge files
    create_sample_files()
    
    # Load files into vector database
    files_to_load = [
        'ai_knowledge.txt',
        'company_data.csv',
        'product_catalog.json'
    ]
    
    print("\n📚 Loading knowledge base...")
    total_docs = 0
    for file_path in files_to_load:
        try:
            count = loader.load_file(file_path)
            total_docs += count
            print(f"✓ Loaded {count} documents from {file_path}")
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
    
    print(f"\n📊 Total documents in knowledge base: {total_docs}")
    
    # Initialize AI Agent
    agent = AIAgent(loader)
    
    # Example queries
    example_queries = [
        "What is machine learning?",
        "Who works in the engineering department?",
        "What electronics products are available?",
        "Tell me about neural networks",
        "What is the price of the smartphone?"
    ]
    
    print("\n🔍 EXAMPLE QUERIES")
    print("=" * 50)
    
    for query in example_queries:
        print(f"\n{agent.query(query)}")
        print("\n" + "=" * 50)


def create_sample_files():
    """Create sample files for demonstration."""
    
    # AI Knowledge base
    ai_content = """
    Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.

    Deep Learning is a specialized area of machine learning that uses neural networks with multiple layers (hence "deep") to model and understand complex patterns in data.

    Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information.

    Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language in a valuable way.

    Computer Vision enables machines to interpret and understand visual information from the world, such as images and videos.

    Reinforcement Learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward.
    """
    
    with open('ai_knowledge.txt', 'w') as f:
        f.write(ai_content)
    
    # Company data CSV
    import csv
    company_data = [
        {'name': 'John Smith', 'department': 'Engineering', 'role': 'Senior Developer', 'location': 'New York'},
        {'name': 'Sarah Johnson', 'department': 'Marketing', 'role': 'Marketing Manager', 'location': 'California'},
        {'name': 'Mike Wilson', 'department': 'Engineering', 'role': 'DevOps Engineer', 'location': 'Texas'},
        {'name': 'Emily Davis', 'department': 'Sales', 'role': 'Sales Representative', 'location': 'Florida'},
        {'name': 'David Brown', 'department': 'Engineering', 'role': 'Data Scientist', 'location': 'Washington'}
    ]
    
    with open('company_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'department', 'role', 'location'])
        writer.writeheader()
        writer.writerows(company_data)
    
    # Product catalog JSON
    product_catalog = {
        "electronics": [
            {
                "id": "ELEC001",
                "name": "Smartphone Pro",
                "category": "Mobile Devices",
                "price": 899.99,
                "features": ["5G connectivity", "Triple camera", "Fast charging"],
                "in_stock": True
            },
            {
                "id": "ELEC002", 
                "name": "Wireless Headphones",
                "category": "Audio",
                "price": 199.99,
                "features": ["Noise cancellation", "30-hour battery", "Bluetooth 5.0"],
                "in_stock": True
            },
            {
                "id": "ELEC003",
                "name": "Gaming Laptop",
                "category": "Computers",
                "price": 1499.99,
                "features": ["RTX Graphics", "16GB RAM", "1TB SSD"],
                "in_stock": False
            }
        ],
        "books": [
            {
                "id": "BOOK001",
                "name": "AI for Beginners",
                "category": "Education",
                "price": 29.99,
                "author": "Dr. Jane Doe",
                "pages": 350
            }
        ]
    }
    
    with open('product_catalog.json', 'w') as f:
        json.dump(product_catalog, f, indent=2)


if __name__ == "__main__":
    demonstrate_ai_agent()