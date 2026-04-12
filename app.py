import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from memory_compression import should_compress, compress_memory, build_history, update_memory

# Load env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant with memory.


- You MUST use the provided conversation history.
- If the user has shared personal information (like name, preferences, etc.), you MUST remember and use it.
- DO NOT say "I don't remember" or "I don't have memory".
- Always check history before answering.

Conversation History:
{history}

User: {input}
AI:
""")


# Chain (modern LangChain)
chain = prompt | llm

print("🚀 AI Chatbot Started (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    try:
        response = chain.invoke({
            "history": build_history(user_input),
            "input": user_input
        })

        reply = response.content
        print("Bot:", reply)

        # Update memory
        update_memory(user_input, reply)
        if should_compress():
            compress_memory()
    except Exception as e:
        print("⚠️ Error:", e)