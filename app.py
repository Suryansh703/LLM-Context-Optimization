import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

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
You are a smart and helpful assistant. Answer the user's question based on the conversation history.

Conversation:
{history}

User: {input}
AI:
""")

# Memory (manual for now)
chat_history = ""

# Chain (modern LangChain)
chain = prompt | llm

print("🚀 AI Chatbot Started (type 'exit' to quit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    try:
        response = chain.invoke({
            "history": chat_history,
            "input": user_input
        })

        reply = response.content
        print("Bot:", reply)

        # Update memory
        chat_history += f"\nUser: {user_input}\nAI: {reply}"

    except Exception as e:
        print("⚠️ Error:", e)