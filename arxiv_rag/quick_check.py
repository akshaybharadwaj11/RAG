# ============================================================================
# 4. QUICK VALIDATION SCRIPT - quick_check.py
# ============================================================================

# Run this for quick sanity check
from validate_ingestion import IngestionValidator
from explore_data import DataExplorer

print("Running quick validation...")

# 1. Validate ingestion
validator = IngestionValidator()
results = validator.run_full_validation()


print("============ Validate Ingestion complete=============")

# 2. Show sample data
if results["overall_status"] == "HEALTHY":
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    
    explorer = DataExplorer()
    samples = explorer.get_sample_chunks(n=2)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"Paper: {sample['paper_title']}")
        print(f"Chunk: {sample['chunk_id']}")
        print(f"Text: {sample['text_preview']}\n")
    
    # 3. Test a query
    print("="*60)
    print("TEST QUERY")
    print("="*60)
    
    from src.vectorstore.embeddings import EmbeddingGenerator
    
    embedding_gen = EmbeddingGenerator()
    query = "attention mechanisms in neural networks"
    query_embedding = embedding_gen.generate_embeddings([query])
    
    results = explorer.collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )
    
    print(f"\nQuery: '{query}'")
    print(f"Results: {len(results['ids'][0])} documents\n")
    
    for i in range(len(results['ids'][0])):
        print(f"{i+1}. {results['metadatas'][0][i]['title']}")
        print(f"   Similarity: {1 - results['distances'][0][i]:.3f}")
        print(f"   Preview: {results['documents'][0][i][:100]}...\n")
