import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

chunks_file = os.path.join("data", "chunks.json")
with open(chunks_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunks_map = {c["chunk_id"]: c for c in chunks}
print("=== chunk_052 ===")
print(chunks_map["chunk_052"]["text"])
print("=== chunk_053 ===")
print(chunks_map["chunk_053"]["text"])
