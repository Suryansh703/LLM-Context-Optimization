import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from memory_compression import (
    should_compress,
    compress_memory,
    build_context,
    update_memory,
    count_ltm_tokens   
)

# ── ENV ──
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ── LLM ──
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# ── PROMPT ──
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant with long-term memory.

Use the structured memory carefully.

User Facts:
{facts}

User Preferences:
{preferences}

User Goals:
{goals}

Summary of Past Interactions:
{summary}

Archived Insights:
{archived}

Recent Conversation:
{recent}

Instructions:
- Prioritize facts and preferences
- Maintain consistency
- Use summary for long-term context
- Use archived insights only if highly relevant
- Avoid contradictions

User: {input}
AI:
""")

chain = prompt | llm

print("🚀 AI Chatbot Started (type 'exit' to quit)")

turn = 0
DEBUG = False

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    try:
        turn += 1
        print(f"\n[Turn {turn}]")

        # 🔥 Memory before compression
        before = count_ltm_tokens()

        # 🔥 Compression step
        if should_compress():
            print("[App] Compression Triggered")
            compress_memory()

        after = count_ltm_tokens()
        print(f"[App] Memory tokens: {before} → {after}")

        # 🔥 Build structured context
        context = build_context(user_input)

        # Safety fallback
        for key in context:
            if not context[key]:
                context[key] = "None"

        # Debug mode
        if DEBUG:
            print("\n[DEBUG CONTEXT]")
            for k, v in context.items():
                print(f"{k}:\n{v}\n")

        # 🔥 Generate response
        response = chain.invoke({
            "facts": context["facts"],
            "preferences": context["preferences"],
            "goals": context["goals"],
            "summary": context["summary"],
            "archived": context["archived"],
            "recent": context["recent"],
            "input": user_input
        })

        reply = response.content
        print("Bot:", reply)

        # 🔥 Update memory AFTER response
        update_memory(user_input, reply)

    except Exception as e:
        print("⚠️ Error:", e)