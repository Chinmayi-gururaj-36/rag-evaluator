import os
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

class Generator:
    """
    A class that handles answer generation using the Groq API.
    It builds a strict context-based prompt and queries a specified LLM
    to answer questions using only the provided document chunks.
    """

    def __init__(self, api_key: str = None, model: str = "llama-3.1-8b-instant") -> None:
        """
        Initializes the Generator by loading the Groq API key and instantiating the client.

        Args:
            api_key: Optional API key. If not provided, it will search for the
                     GROQ_API_KEY environment variable.
            model: The identifier of the Groq-supported LLM.

        Raises:
            ValueError: If the Groq API key cannot be resolved.
        """
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Groq API key not found. Please set the GROQ_API_KEY environment variable "
                "in a .env file or pass it directly to the Generator constructor."
            )

        self.client = Groq(api_key=resolved_key)
        self.model = model

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an answer to a question using the retrieved chunks as context.

        Args:
            question: The user's question to answer.
            retrieved_chunks: A list of dictionaries representing retrieved chunks,
                             where each dictionary must contain 'chunk_id' and 'text'.

        Returns:
            A dictionary containing:
                - answer (str): The generated response from the LLM.
                - question (str): The original question asked.
                - context_chunk_ids (list): The list of chunk IDs used for context.
                - raw_context (str): The raw concatenated context sent to the LLM.

        Raises:
            RuntimeError: If the Groq API call fails.
        """
        # Format the context text and collect chunk IDs
        context_parts = []
        context_chunk_ids = []
        for chunk in retrieved_chunks:
            chunk_id = chunk.get("chunk_id", "Unknown")
            text = chunk.get("text", "")
            context_chunk_ids.append(chunk_id)
            context_parts.append(f"--- Chunk ID: {chunk_id} ---\n{text}")

        raw_context = "\n\n".join(context_parts)

        # Query the LLM using the chat completions endpoint with strict system constraints
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict QA assistant. Answer the user's question using ONLY the "
                            "facts directly mentioned in the provided Context. "
                            "Do NOT use any external knowledge, assumptions, or extrapolate. "
                            "If the Context does not contain the answer, reply EXACTLY with: "
                            "\"This is not covered in the provided material.\" "
                            "Be concise and precise. If the answer involves mathematical formulas "
                            "or equations, reproduce them exactly as they are written in the Context."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{raw_context}\n\nQuestion: {question}"
                    }
                ],
                temperature=0.0  # Set to 0.0 to maximize factual consistency and minimize hallucinations
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}")

        return {
            "answer": answer,
            "question": question,
            "context_chunk_ids": context_chunk_ids,
            "raw_context": raw_context
        }

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Query the RAG generator.")
    parser.add_argument("question", type=str, help="The question to ask.")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant", help="Groq model to use.")
    args = parser.parse_args()

    # Import Retriever dynamically to keep imports organized
    try:
        from rag_pipeline.retriever import Retriever
    except ImportError:
        from retriever import Retriever

    try:
        # Initialize retriever and query it
        retriever = Retriever()
        chunks = retriever.retrieve(args.question, top_k=3)

        # Initialize generator and generate the answer
        generator = Generator(model=args.model)
        result = generator.generate(args.question, chunks)

        print("\n" + "=" * 80)
        print(f"QUESTION: {result['question']}")
        print(f"RETRIEVED CHUNK IDS: {result['context_chunk_ids']}")
        print("=" * 80)
        print(f"ANSWER:\n{result['answer']}")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
