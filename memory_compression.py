# memory_compression.py

# -------------------------------
# GLOBAL MEMORY STRUCTURES
# -------------------------------

short_term_memory = []   # recent chat (list of messages)
long_term_memory = ""    # compressed summary (string)

# -------------------------------
# CONFIG
# -------------------------------

MAX_STM = 6                 # number of recent messages to keep
TOKEN_LIMIT = 1500          # threshold for compression


# TOKEN COUNT 

def count_tokens(text):
    if not text:
        return 0
    return len(text.split())   

# IMPORTANCE CHECK (Optional Enhancement)


IMPORTANT_KEYWORDS = ["name", "project", "goal", "interest"]

def is_important(text):
    return any(word in text.lower() for word in IMPORTANT_KEYWORDS)



# SHOULD COMPRESS (KEEP SAME NAME)


def should_compress():
    global long_term_memory
    return count_tokens(long_term_memory) > TOKEN_LIMIT



# COMPRESS MEMORY (LLM-BASED)


def compress_memory():
    global long_term_memory

    if not long_term_memory.strip():
        return

    try:
        # Lazy import to avoid circular dependency
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

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

        response = llm.invoke(summary_prompt)
        long_term_memory = response.content.strip()

    except Exception as e:
        print("⚠️ Compression Error:", e)



# UPDATE MEMORY (HELPER FUNCTION)


def update_memory(user_input, ai_output):
    global short_term_memory, long_term_memory

    # Add new messages to STM
    short_term_memory.append(f"User: {user_input}")
    short_term_memory.append(f"AI: {ai_output}")

    # If STM exceeds limit → move old messages to LTM
    if len(short_term_memory) > MAX_STM:
        overflow = short_term_memory[:-MAX_STM]

        # Add overflow to long-term memory
        long_term_memory += "\n" + "\n".join(overflow)

        # Keep only recent messages
        short_term_memory = short_term_memory[-MAX_STM:]



# BUILD HISTORY (USED IN MAIN CODE)


def build_history():
    global short_term_memory, long_term_memory

    history = ""

    if long_term_memory.strip():
        history += f"Long Term Memory:\n{long_term_memory}\n\n"

    if short_term_memory:
        history += "Recent Conversation:\n"
        history += "\n".join(short_term_memory)

    return history.strip()