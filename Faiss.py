import os
import json
import uuid
import time
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM   = 768

# 🔥 Reduced retrieval (compression-first design)
TOP_K = 1

STORE_DIR  = Path("./faiss_store")
INDEX_FILE = STORE_DIR / "index.bin"
META_FILE  = STORE_DIR / "metadata.json"

STORE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────
def _load_index():
    if INDEX_FILE.exists():
        return faiss.read_index(str(INDEX_FILE))
    return faiss.IndexFlatL2(EMBEDDING_DIM)


def _load_meta():
    if META_FILE.exists():
        with open(META_FILE) as f:
            return json.load(f)
    return {"id_map": [], "entries": {}}


def _save(index, meta):
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────
def _embed(text: str, mode="document"):
    try:
        task_type = "retrieval_document" if mode == "document" else "retrieval_query"

        result = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=text
)

        return result.embeddings[0].values

    except Exception as e:
        print(f"[FAISS] Embedding failed: {e}")
        return None

# ─────────────────────────────────────────────
# STORE COMPRESSED MEMORY
# ─────────────────────────────────────────────
def store_summary(summary_text: str):

    if not summary_text.strip():
        return

    index = _load_index()
    meta  = _load_meta()

    # avoid duplicates
    existing = [entry["summary"] for entry in meta["entries"].values()]
    if summary_text in existing:
        return

    embedding = _embed(summary_text, mode="document")
    if embedding is None:
        return

    mem_id = str(uuid.uuid4())
    vector = np.array([embedding], dtype=np.float32)

    index.add(vector)

    meta["id_map"].append(mem_id)
    meta["entries"][mem_id] = {
        "id": mem_id,
        "summary": summary_text,
        "timestamp": time.time()
    }

    _save(index, meta)

    print(f"[FAISS] Stored compressed snapshot | total: {index.ntotal}")

# ─────────────────────────────────────────────
# RETRIEVE COMPRESSED ARCHIVAL INSIGHTS (NOT RAG)
# ─────────────────────────────────────────────
def retrieve_relevant(query: str):

    index = _load_index()
    meta  = _load_meta()

    if index.ntotal == 0:
        return ""

    embedding = _embed(query, mode="query")
    if embedding is None:
        return ""

    k = min(TOP_K, index.ntotal)

    query_vector = np.array([embedding], dtype=np.float32)

    _, indices = index.search(query_vector, k)

    blocks = []

    current_time = time.time()
    MAX_AGE = 60 * 60 * 24  # 1 day

    for idx in indices[0]:

        if idx == -1:
            continue

        mem_id = meta["id_map"][idx]
        entry  = meta["entries"].get(mem_id)

        if entry:

            # 🔥 time-based filtering (research feature)
            if current_time - entry["timestamp"] < MAX_AGE:

                # 🔥 limit size (avoid context explosion)
                blocks.append(entry["summary"][:200])

    result = "\n".join(blocks)

    if result:
        print(f"[FAISS] Retrieved {len(blocks)} compressed insights")

    return result

