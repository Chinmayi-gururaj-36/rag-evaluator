import os
import sys
import json
import time

def main():
    chunks_path = os.path.join("data", "chunks.json")
    vector_store_dir = os.path.join("rag_pipeline", "vector_store")
    index_path = os.path.join(vector_store_dir, "chunk_index.faiss")
    metadata_path = os.path.join(vector_store_dir, "chunk_metadata.json")

    # Basic error handling: check if chunks.json exists
    if not os.path.exists(chunks_path):
        print(f"Error: The chunks file '{chunks_path}' was not found. Please ensure it exists.", file=sys.stderr)
        sys.exit(1)

    print("Loading chunks...")
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing chunks.json: {e}", file=sys.stderr)
        sys.exit(1)

    total_chunks = len(chunks)
    print(f"Loaded {total_chunks} chunks.")

    # Lazy import to ensure script loads quickly and prints target error early if missing dependencies
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except ImportError as e:
        print(f"Dependency error: {e}", file=sys.stderr)
        print("Please ensure you have installed the requirements using: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    print("Initializing SentenceTransformer ('all-MiniLM-L6-v2')...")
    start_time = time.time()
    
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    print("Generating embeddings for all chunks...")
    texts = [chunk["text"] for chunk in chunks]
    
    try:
        embeddings = model.encode(texts, show_progress_bar=True)
    except Exception as e:
        print(f"Error generating embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]

    print(f"Building FAISS Index (dimension = {dimension})...")
    try:
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
    except Exception as e:
        print(f"Error building FAISS index: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(vector_store_dir, exist_ok=True)

    print("Saving FAISS index and metadata...")
    try:
        faiss.write_index(index, index_path)
    except Exception as e:
        print(f"Error saving FAISS index to '{index_path}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Write metadata in same order as the index
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving metadata to '{metadata_path}': {e}", file=sys.stderr)
        sys.exit(1)

    end_time = time.time()
    time_taken = end_time - start_time

    # Printed stats as required
    print("\nEmbedding and Indexing Completed Successfully!")
    print(f"  - Total chunks embedded: {total_chunks}")
    print(f"  - Embedding dimension: {dimension}")
    print(f"  - Time taken: {time_taken:.2f} seconds")
    print(f"  - FAISS index saved to: {index_path}")
    print(f"  - Chunk metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
