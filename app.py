import os
from dotenv import load_dotenv
from google import genai


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conversation = []
compressed_memory = ""

THRESHOLD = 6

def compress_memory():
    global conversation, compressed_memory

    prompt = f"""
    Summarise the following conversation.
    Keep only important facts, user preferences, and key details.

    Conversation:
    {conversation}
    """

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        summary = response.text

        compressed_memory += "\n" + summary
        conversation = []

    except Exception as e:
        print("Compression Error:", e)



def chat(user_input):
    global conversation, compressed_memory

    conversation.append(f"User: {user_input}")

    if len(conversation) > THRESHOLD:
        compress_memory()

    final_prompt = f"""
    Previous important memory:
    {compressed_memory}

    Recent conversation:
    {conversation}

    Answer the user properly.
    """

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=final_prompt
        )

        reply = response.text

    except Exception as e:
        print("Error:", e)
        return "⚠️ API error. Try again later."

    conversation.append(f"Bot: {reply}")

    return reply



print("Smart Chatbot Started 🚀 (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    reply = chat(user_input)
    print("Bot:", reply)
