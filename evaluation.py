import random
from memory_compression import *

# -----------------------------
# BASELINE MEMORY (no compression)
# -----------------------------
baseline_memory = []

def baseline_update(user, ai):
    baseline_memory.append(f"User: {user}")
    baseline_memory.append(f"AI: {ai}")

def baseline_tokens():
    return len(" ".join(baseline_memory).split())


# -----------------------------
# TEST DATA
# -----------------------------
facts = [
    "My name is Suryansh",
    "I like Python",
    "I prefer AI",
    "My goal is to become ML engineer"
]

questions = [
    ("What is my name?", "Suryansh"),
    ("What do I like?", "Python"),
    ("What is my goal?", "ML engineer")
]


# -----------------------------
# SIMULATION
# -----------------------------
def run_experiment(turns=100):

    print("\n🚀 Running Experiment...\n")

    # reset memory
    global short_term_memory, long_term_memory
    short_term_memory = []
    long_term_memory = {
        "facts": [],
        "preferences": [],
        "goals": [],
        "summary": ""
    }

    baseline_memory.clear()

    growth_tokens = []

    for i in range(turns):

        # Inject facts early
        if i < 5:
            user_input = random.choice(facts)
        else:
            user_input = "random conversation text " + str(i)

        ai_output = "ok"

        # baseline update
        baseline_update(user_input, ai_output)

        # your model update
        update_memory(user_input, ai_output)

        if should_compress():
            compress_memory()

        growth_tokens.append(count_ltm_tokens())

    # -----------------------------
    # METRICS
    # -----------------------------

    # 1. Accuracy
    correct = 0

    for q, ans in questions:
        context = build_context(q)

        if ans.lower() in (
            context["facts"].lower() +
            context["summary"].lower()
        ):
            correct += 1

    accuracy = correct / len(questions)

    # 2. Tokens
    baseline = baseline_tokens()
    compressed = count_ltm_tokens()

    # 3. Compression %
    reduction = ((baseline - compressed) / baseline) * 100

    print("📊 RESULTS")
    print("-------------------------")
    print(f"Baseline Tokens: {baseline}")
    print(f"Compressed Tokens: {compressed}")
    print(f"Reduction %: {reduction:.2f}%")
    print(f"Retention Accuracy: {accuracy:.2f}")

    return growth_tokens

## GRAPH
import matplotlib.pyplot as plt

tokens = run_experiment(100)

plt.plot(tokens)
plt.xlabel("Conversation Turns")
plt.ylabel("Memory Tokens")
plt.title("Memory Growth Curve")
plt.show()
