import os
import json
import uuid
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIM   = 768
TOP_K           = 3
STORE_DIR       = Path("./faiss_store")
INDEX_FILE      = STORE_DIR / "index.bin"
META_FILE       = STORE_DIR / "metadata.json"

STORE_DIR.mkdir(exist_ok=True)


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


def _embed(text: str):
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        print(f"[VectorStore] Embedding failed: {e}")
        return None


def store_summary(summary_text: str):
    """
    Call this after compress_memory() runs.
    Embeds the compressed summary and stores it in FAISS.
    """
    if not summary_text.strip():
        return

    index     = _load_index()
    meta      = _load_meta()
    embedding = _embed(summary_text)

    if embedding is None:
        return

    mem_id = str(uuid.uuid4())
    vector = np.array([embedding], dtype=np.float32)

    index.add(vector)
    meta["id_map"].append(mem_id)
    meta["entries"][mem_id] = {
        "id":      mem_id,
        "summary": summary_text
    }

    _save(index, meta)
    print(f"[VectorStore] Summary stored. Total in DB: {index.ntotal}")


def retrieve_relevant(query: str) -> str:
    """
    Call this in build_history() to get relevant past summaries.
    Returns a formatted string ready to inject into the prompt.
    """
    index = _load_index()
    meta  = _load_meta()

    if index.ntotal == 0:
        return ""

    embedding = _embed(query)
    if embedding is None:
        return ""

    k            = min(TOP_K, index.ntotal)
    query_vector = np.array([embedding], dtype=np.float32)
    _, indices   = index.search(query_vector, k)

    blocks = []
    for i, idx in enumerate(indices[0], 1):
        if idx == -1:
            continue
        mem_id = meta["id_map"][idx]
        entry  = meta["entries"].get(mem_id)
        if entry:
            blocks.append(f"[Retrieved Memory {i}]\n{entry['summary']}")

    result = "\n\n".join(blocks)
    if result:
        print(f"[VectorStore] Retrieved {len(blocks)} relevant memories.")
    return result