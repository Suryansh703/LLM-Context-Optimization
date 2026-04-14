import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from memory_compression import (
    should_compress,
    compress_memory,
    build_context,
    update_memory
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
- Use archived insights only if relevant
- Avoid contradictions

User: {input}
AI:
""")

chain = prompt | llm

print("🚀 AI Chatbot Started (type 'exit' to quit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    try:
        # Step 1: Compression BEFORE response
        if should_compress():
            compress_memory()

        # Step 2: Build context
        context = build_context(user_input)

        # Safety fallback
        for key in context:
            if not context[key]:
                context[key] = "None"

        # Debug
        print("\n[DEBUG] Context:")
        print(context)
        print()

        # Step 3: Generate response
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

        # Step 4: Update memory
        update_memory(user_input, reply)

    except Exception as e:
        print("⚠️ Error:", e)
        