
#!/usr/bin/env python3
"""
Self-contained RAG retrieval test.

Run:
    python score_test.py

Optional extra questions:
    python score_test.py "who is An?" "where is the retreat?"

Install:
    pip install lancedb sentence-transformers

Uses:
    multilingual-e5-small
    CPU embeddings
    Temporary LanceDB database
"""

import re
import sys
import tempfile

import lancedb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL = "intfloat/multilingual-e5-small"

MAX_WORDS = 100
OVERLAP_SENTENCES = 1
TOP_SHOW = 3

DOC_TEXT = """Zentora Labs Company Wiki

About the company
Zentora Labs was founded in 2019 by Linh Tran, who still serves as chief executive officer. The company builds privacy-focused software and is headquartered in Da Nang, Vietnam. As of November 2026 it has 48 employees: 30 in Da Nang, 12 in Ho Chi Minh City and 6 in Singapore. A different person, Linh Pham, leads human resources and is not related to the founder.

Products
The main product is QuillBase, an encrypted note-taking app available on iOS, Android, Windows and macOS. QuillBase does not support Linux desktop yet; a Linux version is planned for the second quarter of 2027. An older product called QuillLite was discontinued in 2024 and is no longer sold or supported. Version 3.2 of QuillBase was released on 14 August 2026 and introduced offline mode, which lets users edit notes without an internet connection and sync them later.

Pricing
QuillBase has three plans. The Free plan is limited to 50 notes. The Pro plan costs 6 US dollars per month, or 20 percent less when billed annually. The Team plan costs 12 US dollars per user per month and requires at least 5 users. Education customers receive a 30 percent discount on the Pro plan but not on the Team plan.

Engineering team
An is a backend engineer based in Da Nang. He moved there from Ho Chi Minh City in 2023 and reports to Cuong, who leads the data and platform team. Cuong's favorite programming language is Rust, and under his direction the team rewrote the sync service in Rust during 2025. The rewrite reduced average sync time from 900 milliseconds to 210 milliseconds.

Binh heads the design department. She has worked at Zentora for six years and designed the current QuillBase interface. Her team of four designers works closely with the mobile engineers, and she reviews every release before it ships.

Dung is the finance manager. She joined in March 2022 after several years at a regional bank. She prepares the quarterly budget and approves all purchases above 2,000 US dollars. Purchases below that amount only need approval from a team lead.

Security
Hoa is the security engineer and the company's designated security officer. She owns a cat named Mochi, which often appears in the team's video calls. All employees must use passwords of at least 14 characters and enable two-factor authentication on every company account. Suspected incidents must be reported to Hoa immediately.

On 3 September 2026 QuillBase suffered an outage that lasted 47 minutes. The cause was an expired TLS certificate on the sync servers. Hoa renewed the certificate and added automatic expiry alerts so that the problem cannot happen silently again. No user data was lost during the outage.

Policies
Full-time employees receive 18 days of paid annual leave per year, and up to 5 unused days can be carried over into the next year. Sick leave is separate and allows 10 paid days; a doctor's note is required after 3 consecutive days. Remote work is allowed up to 2 days per week in Da Nang and Ho Chi Minh City, but only 1 day per week in Singapore because of local office rules. Remote workers receive an internet allowance of 500,000 VND per month.

Office hours are 9:00 to 17:30, Monday to Friday. The Singapore office closes later, at 18:00. The annual company retreat takes place every October on Phu Quoc island, and attendance is optional but encouraged.

Roadmap
In the first quarter of 2027 the team plans to ship shared notebooks for the Team plan. The second quarter of 2027 is reserved for the Linux desktop app. A public API is planned for the third quarter of 2027, and its design is led by An together with Cuong.
"""


# (question, expected text, type)
QUESTIONS = [
    ("Who founded Zentora Labs?", "founded in 2019", "in"),
    ("What caused the 3 September 2026 outage?", "expired TLS certificate", "in"),
    ("How much does the Pro plan cost?", "6 US dollars", "in"),
    ("Where did Dung work before Zentora?", "regional bank", "in"),
    ("Who designed the current QuillBase interface?", "designed the current QuillBase interface", "in"),
    ("Who does An report to?", "reports to Cuong", "in"),
    ("How many employees are in Singapore?", "6 in Singapore", "in"),
    ("By how much did sync time improve after the Rust rewrite?", "210 milliseconds", "in"),
    ("What purchase amount needs the finance manager's approval?", "2,000 US dollars", "in"),
    ("How many remote days are allowed in Singapore?", "only 1 day per week", "in"),
    ("Does QuillBase support Linux today?", "does not support Linux", "in"),
    ("Can education customers get a discount on the Team plan?", "Education customers", "in"),
    ("Is QuillLite still sold?", "QuillLite was discontinued", "in"),
    ("Who leads human resources?", "Linh Pham", "in"),
    ("Who leads the team that rewrote the sync service?", "rewrote the sync service", "in"),
    ("Who is leading the public API design?", "public API", "in"),
    ("Who fixed the outage?", "renewed the certificate", "in"),
    ("What is Zentora's annual revenue?", None, "out"),
    ("What is Cuong's phone number?", None, "out"),
    ("Who is Bill Gates?", None, "out"),
    ("What is the capital of France?", None, "out"),
]


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def chunk_text(text, max_words=MAX_WORDS):
    """
    Section-aware chunking.

    - Keeps section headings with their content.
    - Splits long paragraphs at sentence boundaries.
    - Adds sentence overlap to preserve context.
    """

    paragraphs = [
        " ".join(p.split())
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    pending_heading = ""

    for para in paragraphs:

        # Treat short lines without sentence punctuation as headings.
        is_heading = (
            len(para.split()) <= 6
            and not re.search(r"[.!?]$", para)
        )

        if is_heading:
            pending_heading = (
                pending_heading + " " + para
            ).strip()
            continue

        if pending_heading:
            para = pending_heading + ". " + para
            pending_heading = ""

        sentences = re.split(r"(?<=[.!?])\s+", para)

        current = []

        for sentence in sentences:

            candidate = current + [sentence]

            if current and len(" ".join(candidate).split()) > max_words:

                chunks.append(" ".join(current))

                # Preserve the previous sentence as context.
                current = (
                    current[-OVERLAP_SENTENCES:]
                    if OVERLAP_SENTENCES > 0
                    else []
                )

            current.append(sentence)

        if current:
            chunks.append(" ".join(current))

    if pending_heading:
        chunks.append(pending_heading)

    return chunks


def short(text, n=100):
    return text if len(text) <= n else text[:n - 3] + "..."


# --------------------------------------------------
# Main retrieval test
# --------------------------------------------------

def main():

    print("\n=== RAG Retrieval Benchmark ===", flush=True)

    # 1. Chunk document
    chunks = chunk_text(DOC_TEXT)

    sizes = [len(c.split()) for c in chunks]

    print(
        f"\nDocument: {len(chunks)} chunks, "
        f"{min(sizes)}-{max(sizes)} words each",
        flush=True
    )

    # 2. Load embedding model
    print(f"\nLoading model: {MODEL}", flush=True)
    print("Using CPU. First run may download the model...", flush=True)

    model = SentenceTransformer(MODEL, device="cpu")

    # E5 requires passage: prefix for documents.
    passages = ["passage: " + chunk for chunk in chunks]

    print("\nEmbedding document chunks...", flush=True)

    doc_vectors = model.encode(
        passages,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # 3. Create temporary LanceDB database
    print("Creating temporary LanceDB table...", flush=True)

    db = lancedb.connect(tempfile.mkdtemp())

    table = db.create_table(
        "wiki",
        data=[
            {
                "text": chunk,
                "vector": vector.tolist()
            }
            for chunk, vector in zip(chunks, doc_vectors)
        ]
    )

    # 4. Prepare questions
    items = list(QUESTIONS)

    # Extra command-line questions are displayed but not scored.
    items += [
        (question, None, "adhoc")
        for question in sys.argv[1:]
    ]

    # E5 requires query: prefix for questions.
    queries = ["query: " + q for q, _, _ in items]

    print(
        f"\nEmbedding {len(queries)} questions...",
        flush=True
    )

    query_vectors = model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # 5. Search
    results = []

    print("\nSearching...", flush=True)

    for (question, expected, kind), query_vector in zip(
        items, query_vectors
    ):

        rows = (
            table.search(query_vector.tolist())
            .metric("cosine")
            .limit(len(chunks))
            .to_list()
        )

        scores = [
            1.0 - row["_distance"]
            for row in rows
        ]

        texts = [
            row["text"]
            for row in rows
        ]

        rank = None

        if expected:
            rank = next(
                (
                    i + 1
                    for i, chunk in enumerate(texts)
                    if expected.lower() in chunk.lower()
                ),
                None
            )

        results.append(
            (question, expected, kind, rank, scores, texts)
        )

    # 6. Display results
    for number, result in enumerate(results, 1):

        question, expected, kind, rank, scores, texts = result

        note = " (not in the document)" if kind == "out" else ""

        print(f"\nQ{number}. {question}{note}")

        for i in range(min(TOP_SHOW, len(texts))):

            mark = "✓" if rank == i + 1 else " "

            print(
                f"   #{i + 1}  "
                f"{scores[i]:.3f} {mark} "
                f"{short(texts[i])}"
            )

        if rank and rank > TOP_SHOW:

            print(
                f"   ... correct answer is #{rank} "
                f"({scores[rank - 1]:.3f}): "
                f"{short(texts[rank - 1])}"
            )

    # --------------------------------------------------
    # 7. Evaluation
    # --------------------------------------------------

    ins = [r for r in results if r[2] == "in"]
    outs = [r for r in results if r[2] == "out"]

    hit1 = sum(1 for r in ins if r[3] == 1)

    hit3 = sum(
        1 for r in ins
        if r[3] is not None and r[3] <= 3
    )

    reciprocal_ranks = [
        1.0 / r[3] if r[3] is not None else 0.0
        for r in ins
    ]

    mrr = (
        sum(reciprocal_ranks) / len(ins)
        if ins else 0.0
    )

    print("\n" + "=" * 45)
    print("              FINAL SCORE")
    print("=" * 45)

    print(f"Hit@1 : {hit1}/{len(ins)} ({hit1 / len(ins) * 100:.1f}%)")
    print(f"Hit@3 : {hit3}/{len(ins)} ({hit3 / len(ins) * 100:.1f}%)")
    print(f"MRR   : {mrr:.3f}")

    if ins and outs:

        low_in = min(r[4][0] for r in ins)
        high_out = max(r[4][0] for r in outs)

        gap = low_in - high_out

        print("\n--- Score gap ---")

        print(
            f"Lowest top-1 score of real questions: "
            f"{low_in:.3f}"
        )

        print(
            f"Highest top-1 score of unrelated questions: "
            f"{high_out:.3f}"
        )

        print(f"Gap: {gap:+.3f}")

        if gap >= 0.10:
            print("Result: A score threshold may separate them.")
        elif gap > 0:
            print("Result: Small gap. A threshold alone is risky.")
        else:
            print("Result: Scores overlap. A threshold alone cannot separate them.")

    print("\nBenchmark completed.\n", flush=True)


if __name__ == "__main__":
    main()
