import os
import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vector_store = FAISS.from_texts(
    ["Initial memory"],
    embedding=embeddings
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a smart AI assistant.

Important memory:
{memory}

Recent conversation:
{history}

User: {input}
AI:
""")

chain = prompt | llm

# memory
chat_history = ""
compressed_memory = ""

THRESHOLD = 6


def get_current_date():
    return datetime.datetime.now().strftime("%A, %d %B %Y")


def compress_memory():

    global chat_history, compressed_memory

    compression_prompt = f"""
    Summarize conversation into key bullet points.

    Conversation:
    {chat_history}
    """

    response = llm.invoke(compression_prompt)

    summary = response.content

    compressed_memory += "\n" + summary

    chat_history = ""


def store_memory(text):

    vector_store.add_texts([text])


def retrieve_memory(query):

    docs = vector_store.similarity_search(query, k=2)

    return "\n".join([doc.page_content for doc in docs])


print("🚀 FAISS Memory Chatbot Started")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    # tool: date
    if "date" in user_input.lower():

        reply = f"Today's date is {get_current_date()}"

    else:

        # compress long conversation
        if len(chat_history.split("\n")) > THRESHOLD:

            compress_memory()

        # retrieve relevant vector memory
        vector_memory = retrieve_memory(user_input)

        response = chain.invoke({

            "memory": compressed_memory + "\n" + vector_memory,

            "history": chat_history,

            "input": user_input

        })

        reply = response.content

    print("Bot:", reply)

    # update memory
    chat_history += f"\nUser: {user_input}\nAI: {reply}"

    store_memory(f"User: {user_input} AI: {reply}")
