"""
RAG (Retrieval-Augmented Generation) Example for Oprel SDK.

Demonstrates querying the local knowledge base.
"""

from oprel.client import OprelClient

def main():
    client = OprelClient()
    
    # 1. Add documents to the knowledge base (simulated here)
    # client.index_directory("./my_documents")
    
    # 2. Search the knowledge base
    results = client.search_knowledge("Oprel architecture")
    print("Search Results:", results)
    
    # 3. The chat API automatically uses the knowledge base if enabled
    # context = "
".join([doc.text for doc in results])
    # ...

if __name__ == "__main__":
    main()
