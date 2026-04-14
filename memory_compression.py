import os
import json
from pathlib import Path
from dotenv import load_dotenv
from Faiss import store_summary, retrieve_relevant

load_dotenv()

# ── Storage ──
MEMORY_FILE = Path("./memory_store/session_memory.json")
MEMORY_FILE.parent.mkdir(exist_ok=True)

# ── Config ──
MAX_STM = 6
TOKEN_LIMIT = 1500

# ── Memory ──
short_term_memory = []

long_term_memory = {
    "facts": [],
    "preferences": [],
    "goals": [],
    "summary": ""
}

# ------------------------
# LOAD / SAVE
# ------------------------

def load_memory():
    global short_term_memory, long_term_memory

    if MEMORY_FILE.exists():
        with open(MEMORY_FILE) as f:
            data = json.load(f)

            short_term_memory = data.get("short_term_memory", [])

            long_term_memory = data.get("long_term_memory", {
                "facts": [],
                "preferences": [],
                "goals": [],
                "summary": ""
            })

        print(f"[Memory] Loaded | STM: {len(short_term_memory)}")

    else:
        print("[Memory] Starting fresh.")


def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump({
            "short_term_memory": short_term_memory,
            "long_term_memory": long_term_memory
        }, f, indent=2)


# ------------------------
# HELPERS
# ------------------------

def count_ltm_tokens():
    return len(json.dumps(long_term_memory).split())


def classify_memory(text):
    text = text.lower()

    if any(x in text for x in ["my name", "i am", "i'm"]):
        return "facts"

    elif any(x in text for x in ["i like", "i love", "i prefer"]):
        return "preferences"

    elif any(x in text for x in ["my goal", "i want", "i aim"]):
        return "goals"

    return "noise"


def get_archived_insights(query):
    try:
        return retrieve_relevant(query)
    except:
        return ""


# ------------------------
# COMPRESSION
# ------------------------

def should_compress():
    return count_ltm_tokens() > TOKEN_LIMIT


def compress_memory():
    global long_term_memory

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        prompt = f"""
You are compressing conversational memory.

Extract ONLY long-term useful information.

Keep:
- Stable user facts
- Preferences
- Goals

Remove:
- Repetition
- Temporary chat
- Greetings

Output STRICT JSON:
{{
    "facts": [],
    "preferences": [],
    "goals": [],
    "summary": "short abstract memory"
}}

Input Memory:
{json.dumps(long_term_memory)}
"""

        response = llm.invoke(prompt)

        compressed = json.loads(response.content)

        # Merge (NOT overwrite)
        long_term_memory["facts"] = list(set(long_term_memory["facts"] + compressed["facts"]))
        long_term_memory["preferences"] = list(set(long_term_memory["preferences"] + compressed["preferences"]))
        long_term_memory["goals"] = list(set(long_term_memory["goals"] + compressed["goals"]))

        long_term_memory["summary"] += "\n" + compressed["summary"]

        # Store snapshot in FAISS
        store_summary(json.dumps(long_term_memory))

        save_memory()

        print("[Memory] Compression complete.")

    except Exception as e:
        print("⚠️ Compression Error:", e)


# ------------------------
# UPDATE MEMORY
# ------------------------

def update_memory(user_input, ai_output):
    global short_term_memory, long_term_memory

    # STM
    short_term_memory.append(f"User: {user_input}")
    short_term_memory.append(f"AI: {ai_output}")

    # Structured LTM
    category = classify_memory(user_input)

    if category != "noise":
        long_term_memory[category].append(user_input)

    # Overflow → summary
    if len(short_term_memory) > MAX_STM:
        overflow = short_term_memory[:-MAX_STM]

        long_term_memory["summary"] += "\n" + "\n".join(overflow)

        short_term_memory = short_term_memory[-MAX_STM:]

    save_memory()


# ------------------------
# BUILD CONTEXT
# ------------------------

def build_context(user_input=""):

    facts = "\n".join(long_term_memory["facts"])
    preferences = "\n".join(long_term_memory["preferences"])
    goals = "\n".join(long_term_memory["goals"])
    summary = long_term_memory["summary"]
    recent = "\n".join(short_term_memory)

    # FAISS fallback (controlled)
    archived = ""
    if len(summary.strip()) < 50 and user_input:
        archived = get_archived_insights(user_input)

    return {
        "facts": facts,
        "preferences": preferences,
        "goals": goals,
        "summary": summary,
        "recent": recent,
        "archived": archived
    }


# Auto-load
load_memory()