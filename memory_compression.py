import os
import json
from pathlib import Path
from dotenv import load_dotenv
from Faiss import store_summary, retrieve_relevant

load_dotenv()

# ── Persistent storage ──
MEMORY_FILE = Path("./memory_store/session_memory.json")
MEMORY_FILE.parent.mkdir(exist_ok=True)

# ── Config ──
MAX_STM            = 6
TOKEN_LIMIT        = 1500
IMPORTANT_KEYWORDS = ["name", "project", "goal", "interest"]

# ── Memory state ──
short_term_memory = []
long_term_memory  = ""


# ─────────────────────────────────────────────
# LOAD & SAVE
# ─────────────────────────────────────────────

def load_memory():
    global short_term_memory, long_term_memory
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE) as f:
            data              = json.load(f)
            short_term_memory = data.get("short_term_memory", [])
            long_term_memory  = data.get("long_term_memory", "")
        print(f"[Memory] Loaded — STM: {len(short_term_memory)} messages | LTM: {count_tokens(long_term_memory)} tokens")
    else:
        print("[Memory] No saved memory found. Starting fresh.")


def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump({
            "short_term_memory": short_term_memory,
            "long_term_memory":  long_term_memory
        }, f, indent=2)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def count_tokens(text):
    if not text:
        return 0
    return len(text.split())


def is_important(text):
    return any(word in text.lower() for word in IMPORTANT_KEYWORDS)


# ─────────────────────────────────────────────
# COMPRESSION
# ─────────────────────────────────────────────

def should_compress():
    return count_tokens(long_term_memory) > TOKEN_LIMIT


def compress_memory():
    global long_term_memory

    if not long_term_memory.strip():
        return

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        summary_prompt = f"""
        Summarize the following conversation.
        Keep:
        - Important facts
        - User preferences
        - Goals
        Remove unnecessary repetition.

        Conversation:
        {long_term_memory}
        """

        response         = llm.invoke(summary_prompt)
        compressed       = response.content.strip()
        long_term_memory = compressed

        store_summary(compressed)
        save_memory()

        print("[Memory] Compression done and saved.")

    except Exception as e:
        print("⚠️ Compression Error:", e)


# ─────────────────────────────────────────────
# UPDATE MEMORY
# ─────────────────────────────────────────────

def update_memory(user_input, ai_output):
    global short_term_memory, long_term_memory

    short_term_memory.append(f"User: {user_input}")
    short_term_memory.append(f"AI: {ai_output}")

    if is_important(user_input):
        long_term_memory += f"\nUser Info: {user_input}"

    if len(short_term_memory) > MAX_STM:
        overflow          = short_term_memory[:-MAX_STM]
        long_term_memory += "\n" + "\n".join(overflow)
        short_term_memory = short_term_memory[-MAX_STM:]

    save_memory()


# ─────────────────────────────────────────────
# BUILD HISTORY
# ─────────────────────────────────────────────

def build_history(user_input: str = "") -> str:
    history = ""

    # Bug fixed — now correctly injects retrieved summaries not long_term_memory
    if user_input:
        retrieved = retrieve_relevant(user_input)
        if retrieved:
            history += f"Relevant Past Memory:\n{retrieved}\n\n"

    # Bug fixed — long_term_memory now appears only once
    if long_term_memory.strip():
        history += f"Long Term Memory:\n{long_term_memory}\n\n"

    if short_term_memory:
        history += "Recent Conversation:\n"
        history += "\n".join(short_term_memory)

    return history.strip()


# Auto-load on import
load_memory()