import os
import json
from pathlib import Path
from dotenv import load_dotenv
from Faiss import store_summary, retrieve_relevant

load_dotenv()

# ─────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────
MEMORY_FILE = Path("./memory_store/session_memory.json")
MEMORY_FILE.parent.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_STM = 6
TOKEN_LIMIT = 1500

# 🔥 Compression Constraints (NEW)
MAX_FACTS = 10
MAX_PREFS = 10
MAX_GOALS = 10
MAX_SUMMARY_TOKENS = 300

# ─────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────
short_term_memory = []

long_term_memory = {
    "facts": [],
    "preferences": [],
    "goals": [],
    "summary": ""
}

# ─────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────
def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump({
            "short_term_memory": short_term_memory,
            "long_term_memory": long_term_memory
        }, f, indent=2)
def load_memory():
    global short_term_memory, long_term_memory

    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE) as f:
                content = f.read().strip()

                # 🔥 Handle empty file
                if not content:
                    raise ValueError("Empty memory file")

                data = json.loads(content)

            short_term_memory = data.get("short_term_memory", [])

            loaded_ltm = data.get("long_term_memory", {})

            if isinstance(loaded_ltm, str):
                long_term_memory = {
                    "facts": [],
                    "preferences": [],
                    "goals": [],
                    "summary": loaded_ltm
                }
            else:
                long_term_memory = loaded_ltm

            print(f"[Memory] Loaded | STM: {len(short_term_memory)}")

        except Exception as e:
            print(f"[Memory] Corrupted file detected. Resetting... ({e})")

            short_term_memory = []
            long_term_memory = {
                "facts": [],
                "preferences": [],
                "goals": [],
                "summary": ""
            }

            save_memory()

    else:
        print("[Memory] Starting fresh.")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def count_tokens(text):
    if not text:
        return 0
    return len(text.split())


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


def build_fallback_compression(memory_data):
    def clean_items(items):
        cleaned = []
        for item in items or []:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned

    facts = clean_items(memory_data.get("facts", []))[:MAX_FACTS]
    preferences = clean_items(memory_data.get("preferences", []))[:MAX_PREFS]
    goals = clean_items(memory_data.get("goals", []))[:MAX_GOALS]

    summary = str(memory_data.get("summary", "") or "").strip()
    if not summary:
        summary_parts = facts + preferences + goals
        summary = " ".join(summary_parts[:8]) if summary_parts else "Compressed memory"
    elif len(summary.split()) > 80:
        summary = " ".join(summary.split()[:80]) + "..."

    return {
        "facts": facts,
        "preferences": preferences,
        "goals": goals,
        "summary": summary,
    }

# ─────────────────────────────────────────────
# 🔥 PRUNING (CORE FEATURE)
# ─────────────────────────────────────────────
def prune_memory():
    global long_term_memory

    # Limit structured memory
    long_term_memory["facts"] = long_term_memory["facts"][-MAX_FACTS:]
    long_term_memory["preferences"] = long_term_memory["preferences"][-MAX_PREFS:]
    long_term_memory["goals"] = long_term_memory["goals"][-MAX_GOALS:]

    # Compress summary if too large
    if count_tokens(long_term_memory["summary"]) > MAX_SUMMARY_TOKENS:

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GEMINI_API_KEY")
            )

            prompt = f"""
Compress this summary further.

Keep only essential long-term information.
Remove redundancy.

Limit output to ~150 tokens.

Summary:
{long_term_memory["summary"]}
"""

            response = llm.invoke(prompt)
            long_term_memory["summary"] = response.content.strip()

        except Exception as e:
            print("⚠️ Pruning compression failed:", e)
            summary_words = long_term_memory["summary"].split()
            if len(summary_words) > MAX_SUMMARY_TOKENS:
                long_term_memory["summary"] = " ".join(summary_words[:MAX_SUMMARY_TOKENS]) + "..."

# ─────────────────────────────────────────────
# COMPRESSION
# ─────────────────────────────────────────────
def should_compress():
    return (
        count_ltm_tokens() > TOKEN_LIMIT
        or count_tokens(long_term_memory["summary"]) > MAX_SUMMARY_TOKENS
    )


def compress_memory():
    global long_term_memory

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import re

        before = count_ltm_tokens()

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

Return STRICT JSON ONLY. No explanations.

Schema:
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

        content = (response.content or "").strip()

        # ❌ Empty response
        if not content:
            raise ValueError("Empty response from LLM")

        # 🔥 Remove markdown/code fences if present
        content = content.replace("```json", "").replace("```", "").strip()

        # 🔥 Try direct JSON parse
        try:
            compressed = json.loads(content)

        except json.JSONDecodeError:
            # 🔥 Try to extract JSON substring
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("No valid JSON found in response")

            json_str = match.group()
            compressed = json.loads(json_str)

        # 🔥 Validate keys (safety)
        if not all(k in compressed for k in ["facts", "preferences", "goals", "summary"]):
            raise ValueError("Invalid JSON schema from LLM")

        # 🔥 Replace (not accumulate)
        long_term_memory["facts"] = list(set(compressed["facts"]))[:MAX_FACTS]
        long_term_memory["preferences"] = list(set(compressed["preferences"]))[:MAX_PREFS]
        long_term_memory["goals"] = list(set(compressed["goals"]))[:MAX_GOALS]
        long_term_memory["summary"] = compressed["summary"]

        # 🔥 Enforce pruning
        prune_memory()

        # Optional archive
        store_summary(json.dumps(long_term_memory))

        save_memory()

        after = count_ltm_tokens()
        print(f"[Compression] Tokens: {before} → {after}")

    except Exception as e:
        import traceback

        print("\n========== COMPRESSION ERROR ==========")
        traceback.print_exc()
        print("=======================================\n")

        print(f"Exception: {repr(e)}")
        print("⚠️ Falling back to lightweight local compression")

        fallback = build_fallback_compression(long_term_memory)

        long_term_memory["facts"] = fallback["facts"]
        long_term_memory["preferences"] = fallback["preferences"]
        long_term_memory["goals"] = fallback["goals"]
        long_term_memory["summary"] = fallback["summary"]

        prune_memory()

        try:
            store_summary(json.dumps(long_term_memory))
        except Exception as faiss_error:
            print(f"⚠️ FAISS storage failed: {faiss_error}")

        save_memory()

# ─────────────────────────────────────────────
# UPDATE MEMORY
# ─────────────────────────────────────────────
def update_memory(user_input, ai_output):
    global short_term_memory, long_term_memory

    # STM
    short_term_memory.append(f"User: {user_input}")
    short_term_memory.append(f"AI: {ai_output}")

    # Classification
    category = classify_memory(user_input)

    if category != "noise":

        if user_input not in long_term_memory[category]:
            long_term_memory[category].append(user_input)

        # immediate pruning
        if category == "facts":
            long_term_memory["facts"] = long_term_memory["facts"][-MAX_FACTS:]

        elif category == "preferences":
            long_term_memory["preferences"] = long_term_memory["preferences"][-MAX_PREFS:]

        elif category == "goals":
            long_term_memory["goals"] = long_term_memory["goals"][-MAX_GOALS:]

    # STM overflow → summary
    if len(short_term_memory) > MAX_STM:
        overflow = short_term_memory[:-MAX_STM]

        long_term_memory["summary"] += "\n" + "\n".join(overflow)

        short_term_memory = short_term_memory[-MAX_STM:]

        # enforce summary control
        if count_tokens(long_term_memory["summary"]) > MAX_SUMMARY_TOKENS:
            prune_memory()

    save_memory()

# ─────────────────────────────────────────────
# BUILD CONTEXT
# ─────────────────────────────────────────────
def build_context(user_input=""):

    facts = "\n".join(long_term_memory["facts"])
    preferences = "\n".join(long_term_memory["preferences"])
    goals = "\n".join(long_term_memory["goals"])
    summary = long_term_memory["summary"]
    recent = "\n".join(short_term_memory)

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

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
load_memory()
if __name__ == "__main__":

    print("Testing compression...")

    update_memory("My name is Suryansh", "Okay")
    update_memory("I like Python", "Nice")
    update_memory("I prefer AI", "Great")

    compress_memory()