import os
import json
import argparse
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    """
    A class that handles vector-based search and retrieval from a FAISS index.
    It loads the SentenceTransformer model, the FAISS index, and the chunk metadata
    once during initialization, providing fast retrieval across repeated queries.
    """

    def __init__(
        self,
        index_path: str = "rag_pipeline/vector_store/chunk_index.faiss",
        metadata_path: str = "rag_pipeline/vector_store/chunk_metadata.json",
        model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        """
        Initializes the Retriever by loading the FAISS index, loading chunk metadata,
        and setting up the SentenceTransformer model.

        Args:
            index_path: Path to the serialized FAISS index file.
            metadata_path: Path to the JSON file containing chunk metadata.
            model_name: Name of the SentenceTransformer model to use for query embedding.

        Raises:
            FileNotFoundError: If the index or metadata files are not found.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)

        # Load the FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Embeds the query, searches the FAISS index, and returns the top_k most similar chunks.

        Args:
            query: The search query string.
            top_k: The number of top documents to retrieve. Must be >= 1.

        Returns:
            A list of dictionaries, where each dictionary contains:
                - chunk_id (str): Unique identifier of the chunk.
                - section (str): Section name.
                - chapter (str): Chapter name.
                - text (str): Full text of the chunk.
                - score (float): Similarity score (L2 distance, smaller is more similar).

        Raises:
            ValueError: If query is empty/whitespace, or top_k is less than 1.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or only whitespace.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Embed the query
        query_embedding = self.model.encode([query])
        query_embedding_np = np.array(query_embedding).astype("float32")

        # Search index
        num_chunks = len(self.metadata)
        actual_k = min(top_k, num_chunks)
        if actual_k == 0:
            return []

        distances, indices = self.index.search(query_embedding_np, actual_k)

        results = []
        for i in range(actual_k):
            idx = indices[0][i]
            if idx == -1:
                continue
            dist = float(distances[0][i])
            chunk = self.metadata[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "chapter": chunk["chapter"],
                "text": chunk["text"],
                "score": dist
            })
        return results

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Query the RAG retriever.")
    parser.add_argument("query", type=str, help="The query to search for.")
    parser.add_argument("--index", type=str, default="rag_pipeline/vector_store/chunk_index.faiss", help="Path to FAISS index.")
    parser.add_argument("--metadata", type=str, default="rag_pipeline/vector_store/chunk_metadata.json", help="Path to metadata.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return.")

    args = parser.parse_args()

    try:
        retriever = Retriever(index_path=args.index, metadata_path=args.metadata)
        results = retriever.retrieve(args.query, top_k=args.top_k)

        print(f"\nResults for query: '{args.query}'")
        print("=" * 80)
        for idx, res in enumerate(results, 1):
            print(f"Rank {idx}:")
            print(f"  chunk_id  : {res['chunk_id']}")
            print(f"  section   : {res['section']}")
            print(f"  chapter   : {res['chapter']}")
            print(f"  score (L2): {res['score']:.4f}")
            text_snippet = res['text'].replace('\n', ' ')
            if len(text_snippet) > 150:
                text_snippet = text_snippet[:150] + "..."
            print(f"  text      : {text_snippet}")
            print("-" * 80)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
