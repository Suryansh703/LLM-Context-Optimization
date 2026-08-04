# import random
# import statistics
# from memory_compression import *

# # -----------------------------
# # BASELINE MEMORY
# # -----------------------------
# baseline_memory = []

# def baseline_update(user, ai):
#     baseline_memory.append(f"User: {user}")
#     baseline_memory.append(f"AI: {ai}")

# def baseline_tokens():
#     return len(" ".join(baseline_memory).split())

# # -----------------------------
# # DATASET
# # -----------------------------
# facts = [
#     "My name is Suryansh",
#     "I like Python",
#     "I prefer AI",
#     "My goal is to become ML engineer"
# ]

# questions = [
#     ("What is my name?", "Suryansh"),
#     ("What do I like?", "Python"),
#     ("What is my goal?", "ML engineer")
# ]

# # -----------------------------
# # METRICS
# # -----------------------------
# def compression_ratio(b, c):
#     return b / c if c else 0

# def token_reduction(b, c):
#     return max(0, ((b - c) / b) * 100) if b else 0

# def retrieval_precision(context, expected):
#     hits = sum(1 for e in expected if e.lower() in context.lower())
#     return hits / len(expected) if expected else 0

# def personalization_accuracy(context, facts):
#     hits = sum(1 for f in facts if f.lower() in context.lower())
#     return hits / len(facts) if facts else 0

# def scalability_index(tokens):
#     return max(tokens) / (sum(tokens)/len(tokens)) if tokens else 0

# def memory_drift_rate(initial, context):
#     lost = sum(1 for f in initial if f.lower() not in context.lower())
#     return lost / len(initial) if initial else 0

# def knowledge_retention(context, expected):
#     hits = sum(1 for e in expected if e.lower() in context.lower())
#     return hits / len(expected) if expected else 0

# def hallucination_reduction(correct, total):
#     return correct / total if total else 0

# def performance_score(acc, reduction):
#     return (0.7 * acc) + (0.3 * (reduction / 100))

# # -----------------------------
# # EFFECTIVE CONTEXT (IMPORTANT FIX)
# # -----------------------------
# def get_context_text(ctx):
#     return (
#         ctx["facts"] +
#         ctx["preferences"] +
#         ctx["goals"] +
#         ctx["summary"]
#     )

# # -----------------------------
# # SINGLE RUN
# # -----------------------------
# def run_experiment(turns=100):

#     global short_term_memory, long_term_memory

#     short_term_memory = []
#     long_term_memory = {
#         "facts": [],
#         "preferences": [],
#         "goals": [],
#         "summary": ""
#     }

#     baseline_memory.clear()

#     growth_tokens = []
#     expected_facts = []

#     for i in range(turns):

#         if i < 5:
#             user_input = random.choice(facts)
#         else:
#             user_input = "random conversation " + str(i)

#         ai_output = "ok"

#         baseline_update(user_input, ai_output)
#         update_memory(user_input, ai_output)

#         # 🔥 Force compression
#         if should_compress() or i % 4 == 0:
#             compress_memory()

#         ctx = build_context("test")
#         context_text = get_context_text(ctx)

#         tokens_now = len(context_text.split())
#         growth_tokens.append(tokens_now)

#         expected_facts.append(user_input)

#     # -----------------------------
#     # FINAL CONTEXT
#     # -----------------------------
#     ctx = build_context("test")
#     context_text = get_context_text(ctx)

#     # -----------------------------
#     # METRICS
#     # -----------------------------
#     baseline = baseline_tokens()
#     compressed = len(context_text.split())

#     reduction = token_reduction(baseline, compressed)
#     ratio = compression_ratio(baseline, compressed)

#     # Accuracy
#     correct = sum(1 for q, ans in questions if ans.lower() in context_text.lower())
#     total = len(questions)

#     acc = correct / total

#     results = {
#         "Compression Ratio": ratio,
#         "Token Reduction (%)": reduction,
#         "Hallucination Reduction": hallucination_reduction(correct, total),
#         "Retrieval Precision": retrieval_precision(context_text, [a for _, a in questions]),
#         "Personalization Accuracy": personalization_accuracy(context_text, facts),
#         "Scalability Index": scalability_index(growth_tokens),
#         "Memory Drift Rate": memory_drift_rate(facts, context_text),
#         "Knowledge Retention": acc,
#         "Performance Score": performance_score(acc, reduction)
#     }

#     return results

# # -----------------------------
# # MULTI RUN (10 TIMES)
# # -----------------------------
# def run_multiple_experiments(runs=10):

#     print("\n🚀 Running 10 Experiments...\n")

#     all_results = {key: [] for key in run_experiment().keys()}

#     for i in range(runs):
#         print(f"Run {i+1}")
#         res = run_experiment()

#         for key in all_results:
#             all_results[key].append(res[key])

#     print("\n📊 FINAL RESULTS (Mean ± Std)\n")
#     print("--------------------------------------------------")

#     for key, values in all_results.items():
#         avg = statistics.mean(values)
#         std = statistics.stdev(values)
#         print(f"{key}: {avg:.3f} ± {std:.3f}")

# # -----------------------------
# # RUN
# # -----------------------------
# if __name__ == "__main__":
#     run_multiple_experiments(10)
 
import os
import random
import statistics
import time

import memory_compression as mc

# ==========================================================
# CONFIGURATION
# ==========================================================

RUNS = int(os.getenv("EVAL_RUNS", "1"))
TURNS = int(os.getenv("EVAL_TURNS", "20"))

# Force compression every N turns during evaluation
FORCE_COMPRESSION_INTERVAL = 10


# ==========================================================
# BASELINE MEMORY (NO COMPRESSION)
# ==========================================================

baseline_memory = []


def baseline_update(user_input, ai_output):
    """
    Stores complete conversation without any compression.
    Used as the baseline.
    """
    baseline_memory.append(f"User: {user_input}")
    baseline_memory.append(f"AI: {ai_output}")


def baseline_token_count():
    """
    Approximate baseline token count.
    """
    return len(" ".join(baseline_memory).split())


def clear_baseline():
    baseline_memory.clear()


# ==========================================================
# RESET MEMORY
# ==========================================================

def reset_project_memory():
    """
    Completely reset the project's memory.
    """

    mc.short_term_memory.clear()

    mc.long_term_memory = {
        "facts": [],
        "preferences": [],
        "goals": [],
        "summary": ""
    }

    if os.path.exists(mc.MEMORY_FILE):
        os.remove(mc.MEMORY_FILE)

    mc.save_memory()


# ==========================================================
# DATASET
# ==========================================================

facts = [
    "My name is Suryansh",
    "I am 21 years old",
    "I live in Kanpur",
    "I study Computer Science",
    "I like Python",
    "I like Java",
    "I enjoy Machine Learning",
    "I prefer AI",
    "I love Cricket",
    "I use VS Code"
]

preferences = [
    "I prefer dark mode",
    "I use macOS",
    "I prefer Gemini",
    "I like backend development",
    "I use LangChain"
]

goals = [
    "My goal is to become ML engineer",
    "I want to publish a research paper",
    "I want to crack Google interview",
    "I want to complete my final year project",
    "I want to learn advanced LLMs"
]

small_talk = [
    "Hello",
    "How are you?",
    "Tell me a joke",
    "Good morning",
    "Nice weather",
    "Can you help me?",
    "Explain recursion",
    "What is AI?",
    "Thank you",
    "Goodbye"
]

questions = [
    ("What is my name?", "Suryansh"),
    ("Where do I live?", "Kanpur"),
    ("Which programming language do I like?", "Python"),
    ("What editor do I use?", "VS Code"),
    ("What is my career goal?", "ML engineer")
]


# ==========================================================
# CONVERSATION GENERATOR
# ==========================================================

def generate_user_message(turn):

    if turn < len(facts):
        return facts[turn]

    elif turn < len(facts) + len(preferences):
        return preferences[turn - len(facts)]

    elif turn < len(facts) + len(preferences) + len(goals):
        return goals[
            turn - len(facts) - len(preferences)
        ]

    return random.choice(small_talk)


# ==========================================================
# MEMORY BUILDERS
# ==========================================================

def get_compressed_memory(context):
    """
    Memory stored after compression.

    Used ONLY for compression metrics.
    """

    return "\n".join([
        context["facts"],
        context["preferences"],
        context["goals"],
        context["summary"]
    ])


def get_chat_context(context):
    """
    Complete context sent to Gemini.

    Used ONLY for retrieval metrics.
    """

    return "\n".join([
        context["facts"],
        context["preferences"],
        context["goals"],
        context["summary"],
        context["recent"],
        context["archived"]
    ])


# ==========================================================
# TOKEN UTILITIES
# ==========================================================

def token_count(text):
    """
    Approximate token count.
    """

    if not text:
        return 0

    return len(text.split())


# ==========================================================
# LATENCY
# ==========================================================

def average_latency(values):

    if len(values) == 0:
        return 0

    return statistics.mean(values)

# ==========================================================
# SINGLE EXPERIMENT
# ==========================================================

def run_experiment(turns=TURNS):
    """
    Runs one complete experiment.
    Returns raw data for evaluation.
    """

    print(f"\nRunning Experiment ({turns} turns)...")

    # ----------------------------------
    # Reset Memories
    # ----------------------------------

    reset_project_memory()
    clear_baseline()

    token_history = []
    compression_latency = []

    # ----------------------------------
    # Simulate Conversation
    # ----------------------------------

    for turn in range(turns):

        user_input = generate_user_message(turn)

        # Dummy assistant reply
        ai_output = "Acknowledged."

        # -------------------------------
        # Baseline Memory
        # -------------------------------

        baseline_update(user_input, ai_output)

        # -------------------------------
        # Update Project Memory
        # -------------------------------

        mc.update_memory(user_input, ai_output)

        # -------------------------------
        # Compression Trigger
        # -------------------------------

        compress = False

        if mc.should_compress():
            compress = True

        elif (turn + 1) % FORCE_COMPRESSION_INTERVAL == 0:
            compress = True

        if compress:

            start = time.perf_counter()

            mc.compress_memory()

            end = time.perf_counter()

            compression_latency.append(end - start)

        # -------------------------------
        # Build Current Context
        # -------------------------------

        context = mc.build_context(user_input)

        chat_context = get_chat_context(context)

        token_history.append(
            token_count(chat_context)
        )

    # =====================================================
    # FINAL MEMORY
    # =====================================================

    final_context = mc.build_context("evaluation")

    # Memory stored after compression
    compressed_memory = get_compressed_memory(
        final_context
    )

    # Complete prompt context
    chat_context = get_chat_context(
        final_context
    )

    # =====================================================
    # TOKEN COUNTS
    # =====================================================

    baseline_tokens = baseline_token_count()

    compressed_tokens = token_count(
        compressed_memory
    )

    chat_tokens = token_count(
        chat_context
    )

    # =====================================================
    # DEBUG
    # =====================================================

    print("\n----------- DEBUG -----------")

    print(f"Baseline Tokens   : {baseline_tokens}")

    print(f"Compressed Tokens : {compressed_tokens}")

    print(f"Chat Tokens       : {chat_tokens}")

    print("-----------------------------\n")

    # =====================================================
    # RETURN RAW DATA
    # =====================================================

    return {

        "baseline_tokens": baseline_tokens,

        "compressed_tokens": compressed_tokens,

        "chat_tokens": chat_tokens,

        "compressed_memory": compressed_memory,

        "chat_context": chat_context,

        "token_history": token_history,

        "compression_latency": compression_latency,

        "questions": questions,

        "facts": facts,

        "preferences": preferences,

        "goals": goals
    }
# ==========================================================
# METRIC FUNCTIONS
# ==========================================================

def compression_ratio(baseline_tokens, compressed_tokens):
    """
    Compression Ratio = Original Memory / Compressed Memory
    Higher is better.
    """

    if compressed_tokens == 0:
        return 0

    return baseline_tokens / compressed_tokens


def token_reduction(baseline_tokens, compressed_tokens):
    """
    Percentage of memory saved.
    """

    if baseline_tokens == 0:
        return 0

    reduction = (
        (baseline_tokens - compressed_tokens)
        / baseline_tokens
    ) * 100

    # Token reduction should never be negative.
    return max(0.0, reduction)


def knowledge_retention(chat_context, questions):
    """
    Can the assistant still remember
    important information?
    """

    correct = 0

    for _, answer in questions:

        if answer.lower() in chat_context.lower():
            correct += 1

    return correct / len(questions)


def retrieval_precision(chat_context,
                        facts,
                        preferences,
                        goals):
    """
    Percentage of important memories
    available in the final prompt.
    """

    expected = facts + preferences + goals

    hits = 0

    for item in expected:

        if item.lower() in chat_context.lower():
            hits += 1

    return hits / len(expected)


def personalization_accuracy(chat_context,
                             facts,
                             preferences,
                             goals):
    """
    Measures whether user-specific
    information survives compression.
    """

    expected = facts + preferences + goals

    matched = 0

    for item in expected:

        if item.lower() in chat_context.lower():
            matched += 1

    return matched / len(expected)


def memory_drift_rate(chat_context,
                      facts,
                      preferences,
                      goals):
    """
    Percentage of memories lost.
    Lower is better.
    """

    expected = facts + preferences + goals

    lost = 0

    for item in expected:

        if item.lower() not in chat_context.lower():
            lost += 1

    return lost / len(expected)


def scalability_index(token_history):
    """
    Memory growth indicator.
    Lower is generally better.
    """

    if len(token_history) == 0:
        return 0

    average = statistics.mean(token_history)

    if average == 0:
        return 0

    return max(token_history) / average


def average_compression_latency(latencies):
    """
    Average compression time.
    """

    if len(latencies) == 0:
        return 0

    return statistics.mean(latencies)


def hallucination_rate(chat_context,
                       facts,
                       preferences,
                       goals):
    """
    Approximate hallucination metric.

    Missing factual memories are treated
    as hallucinated information.
    """

    expected = facts + preferences + goals

    missing = 0

    for item in expected:

        if item.lower() not in chat_context.lower():
            missing += 1

    return missing / len(expected)


def performance_score(retention,
                      reduction):
    """
    Overall project score.
    """

    return (
        (0.7 * retention)
        +
        (0.3 * (reduction / 100))
    )


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate_results(data):
    """
    Compute all metrics for one experiment.
    """

    baseline = data["baseline_tokens"]

    compressed = data["compressed_tokens"]

    chat_context = data["chat_context"]

    retention = knowledge_retention(
        chat_context,
        data["questions"]
    )

    reduction = token_reduction(
        baseline,
        compressed
    )

    results = {

        "Compression Ratio":
            compression_ratio(
                baseline,
                compressed
            ),

        "Token Reduction (%)":
            reduction,

        "Knowledge Retention":
            retention,

        "Retrieval Precision":
            retrieval_precision(
                chat_context,
                data["facts"],
                data["preferences"],
                data["goals"]
            ),

        "Personalization Accuracy":
            personalization_accuracy(
                chat_context,
                data["facts"],
                data["preferences"],
                data["goals"]
            ),

        "Memory Drift Rate":
            memory_drift_rate(
                chat_context,
                data["facts"],
                data["preferences"],
                data["goals"]
            ),

        "Scalability Index":
            scalability_index(
                data["token_history"]
            ),

        "Compression Latency (s)":
            average_compression_latency(
                data["compression_latency"]
            ),

        "Hallucination Rate":
            hallucination_rate(
                chat_context,
                data["facts"],
                data["preferences"],
                data["goals"]
            ),

        "Performance Score":
            performance_score(
                retention,
                reduction
            )

    }

    return results
# ==========================================================
# MULTIPLE EXPERIMENTS
# ==========================================================

def run_multiple_experiments(runs=RUNS):
    """
    Run multiple experiments and report
    Mean ± Standard Deviation.
    """

    print("\n" + "=" * 80)
    print("LLM MEMORY COMPRESSION EVALUATION")
    print("=" * 80)

    all_results = {}

    for run in range(runs):

        print(f"\nExperiment {run + 1}/{runs}")

        experiment_data = run_experiment()

        results = evaluate_results(experiment_data)

        for metric, value in results.items():

            if metric not in all_results:
                all_results[metric] = []

            all_results[metric].append(value)

    print("\n")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\nRuns      : {runs}")
    print(f"Turns/Run : {TURNS}")

    print("\n")

    print("{:<35} {:>15} {:>15}".format(
        "Metric",
        "Mean",
        "Std Dev"
    ))

    print("-" * 70)

    for metric, values in all_results.items():

        mean = statistics.mean(values)

        std = (
            statistics.stdev(values)
            if len(values) > 1
            else 0
        )

        print("{:<35} {:>15.4f} {:>15.4f}".format(
            metric,
            mean,
            std
        ))

    print("-" * 70)

    return all_results


# ==========================================================
# CSV EXPORT
# ==========================================================

def export_results(results,
                   filename="evaluation_results.csv"):
    """
    Export results to CSV.
    """

    import csv

    with open(filename,
              "w",
              newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Metric",
            "Mean",
            "Std Dev"
        ])

        for metric, values in results.items():

            mean = statistics.mean(values)

            std = (
                statistics.stdev(values)
                if len(values) > 1
                else 0
            )

            writer.writerow([
                metric,
                round(mean, 4),
                round(std, 4)
            ])

    print(f"\nResults exported to {filename}")


# ==========================================================
# OPTIONAL DETAILED REPORT
# ==========================================================

def print_summary(results):

    print("\n")
    print("=" * 80)
    print("PROJECT SUMMARY")
    print("=" * 80)

    compression = statistics.mean(
        results["Compression Ratio"]
    )

    reduction = statistics.mean(
        results["Token Reduction (%)"]
    )

    retention = statistics.mean(
        results["Knowledge Retention"]
    )

    retrieval = statistics.mean(
        results["Retrieval Precision"]
    )

    latency = statistics.mean(
        results["Compression Latency (s)"]
    )

    print(f"Compression Ratio      : {compression:.2f}x")
    print(f"Token Reduction        : {reduction:.2f}%")
    print(f"Knowledge Retention    : {retention*100:.2f}%")
    print(f"Retrieval Precision    : {retrieval*100:.2f}%")
    print(f"Compression Latency    : {latency:.3f} sec")

    print("=" * 80)


# ==========================================================
# MAIN
# ==========================================================

def main():

    results = run_multiple_experiments(RUNS)

    export_results(results)

    print_summary(results)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()