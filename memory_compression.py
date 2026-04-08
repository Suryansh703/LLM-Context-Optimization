import os
import re
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

compression_prompt = ChatPromptTemplate.from_template("""
You are a memory compression engine for an AI assistant.
Compress the conversation below into a structured summary.

RULES:
- KEEP: facts, decisions, user preferences, goals, named entities, tasks
- DISCARD: greetings, filler, repetition, small talk

CONVERSATION:
{conversation}

Return ONLY a JSON object, no markdown, no explanation:
{{
  "summary": "2-3 sentence overview",
  "facts": ["fact1", "fact2"],
  "decisions": ["decision1"],
  "user_preferences": ["pref1"],
  "key_entities": ["entity1"]
}}
""")

compression_chain = compression_prompt | llm

chat_history      = ""
compressed_blocks = []
THRESHOLD         = 6
ANCHOR_LINES      = 4


def should_compress() -> bool:
    return chat_history.count("User:") >= THRESHOLD


def compress_memory():
    global chat_history, compressed_blocks

    print(f"\n[Compression] Compressing {chat_history.count('User:')} turns...")

    try:
        response = compression_chain.invoke({"conversation": chat_history})
        raw      = response.content
        cleaned  = re.sub(r"```(?:json)?|```", "", raw).strip()
        summary  = json.loads(cleaned)
        compressed_blocks.append(summary)
        print(f"[Compression] Done. Total summaries: {len(compressed_blocks)}")

    except Exception as e:
        print(f"[Compression] Failed: {e}")
        compressed_blocks.append({
            "summary": chat_history,
            "facts": [], "decisions": [],
            "user_preferences": [], "key_entities": []
        })

    lines        = chat_history.strip().split("\n")
    chat_history = "\n".join(lines[-ANCHOR_LINES:])
    print(f"[Compression] Pruned. Kept last {ANCHOR_LINES} lines.\n")


def format_compressed_memory() -> str:
    if not compressed_blocks:
        return ""

    blocks = []
    for i, mem in enumerate(compressed_blocks, 1):
        lines = [f"[Past Context {i}]"]
        if mem.get("summary"):
            lines.append(f"Summary    : {mem['summary']}")
        if mem.get("facts"):
            lines.append("Facts      : " + " | ".join(mem["facts"]))
        if mem.get("decisions"):
            lines.append("Decisions  : " + " | ".join(mem["decisions"]))
        if mem.get("user_preferences"):
            lines.append("Preferences: " + " | ".join(mem["user_preferences"]))
        if mem.get("key_entities"):
            lines.append("Entities   : " + ", ".join(mem["key_entities"]))
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_history() -> str:
    past   = format_compressed_memory()
    recent = chat_history.strip()

    if past and recent:
        return f"{past}\n\n--- Recent Conversation ---\n{recent}"
    return past or recent