#!/usr/bin/env python3
"""
Self-contained retrieval test. The document and the questions live in this file.
Just run it:

    python score_test.py

It chunks the document, embeds the chunks into a temporary LanceDB table, then
for every question prints the question followed by the top 3 chunks ranked by
score. A check mark (✓) marks the chunk that really holds the answer.
A short summary (hit@1, hit@3, score gap) comes at the end.

Extra questions on the command line are ranked too (shown, not scored):

    python score_test.py "who is An?" "where is the retreat?"

Install: pip install lancedb sentence-transformers
Uses a temp folder each run, so it never touches your app's data.
"""
import re
import sys
import tempfile

import lancedb
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_WORDS = 80      # paragraphs longer than this are cut at sentence boundaries
TOP_SHOW = 3        # chunks shown for failed questions

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

# (question, text that must appear in the chunk holding the answer, or None if NOT in the document)
QUESTIONS = [
    # direct facts
    ("Who founded Zentora Labs?", "founded in 2019"),
    ("What caused the 3 September 2026 outage?", "expired TLS certificate"),
    ("How much does the Pro plan cost?", "6 US dollars"),
    # pronouns: the answer chunk does not repeat the name
    ("Where did Dung work before Zentora?", "regional bank"),
    ("Who designed the current QuillBase interface?", "designed the current QuillBase interface"),
    ("Who does An report to?", "reports to Cuong"),
    # numbers
    ("How many employees are in Singapore?", "6 in Singapore"),
    ("By how much did sync time improve after the Rust rewrite?", "210 milliseconds"),
    ("What purchase amount needs the finance manager's approval?", "2,000 US dollars"),
    ("How many remote days are allowed in Singapore?", "only 1 day per week"),
    # negation
    ("Does QuillBase support Linux today?", "does not support Linux"),
    ("Can education customers get a discount on the Team plan?", "Education customers"),
    ("Is QuillLite still sold?", "QuillLite was discontinued"),
    # confusable names / multi-hop
    ("Who leads human resources?", "Linh Pham"),
    ("Who leads the team that rewrote the sync service?", "rewrote the sync service"),
    ("Who is leading the public API design?", "public API"),
    ("Who fixed the outage?", "renewed the certificate"),
    # NOT in the document
    ("What is Zentora's annual revenue?", None),
    ("What is Cuong's phone number?", None),
    ("Who is Bill Gates?", None),
    ("What is the capital of France?", None),
]


def chunk_text(text, max_words=MAX_WORDS):
    """Paragraph chunks; long paragraphs are cut at sentence boundaries."""
    chunks = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para.split()) <= max_words:
            chunks.append(para)
            continue
        current = []
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if current and len(" ".join(current + [sent]).split()) > max_words:
                chunks.append(" ".join(current))
                current = []
            current.append(sent)
        if current:
            chunks.append(" ".join(current))
    return chunks


def short(text, n=75):
    return text if len(text) <= n else text[: n - 3] + "..."


def main():
    chunks = chunk_text(DOC_TEXT)
    sizes = [len(c.split()) for c in chunks]
    print(f"Document: {len(chunks)} chunks, {min(sizes)}-{max(sizes)} words each")

    print(f"Loading model: {MODEL}")
    model = SentenceTransformer(MODEL)
    vecs = model.encode(chunks, normalize_embeddings=True)
    db = lancedb.connect(tempfile.mkdtemp())
    table = db.create_table(
        "wiki", data=[{"text": c, "vector": v.tolist()} for c, v in zip(chunks, vecs)]
    )

    items = [(q, e, "in" if e else "out") for q, e in QUESTIONS]
    items += [(q, None, "adhoc") for q in sys.argv[1:]]
    qvecs = model.encode([q for q, _, _ in items], normalize_embeddings=True)

    results = []
    for (q, expect, kind), qv in zip(items, qvecs):
        rows = table.search(qv).metric("cosine").limit(len(chunks)).to_list()
        scores = [1.0 - r["_distance"] for r in rows]
        texts = [r["text"] for r in rows]
        rank = None
        if expect:
            rank = next((i + 1 for i, t in enumerate(texts) if expect.lower() in t.lower()), None)
        results.append((q, expect, kind, rank, scores, texts))

    for n, (q, expect, kind, rank, scores, texts) in enumerate(results, 1):
        note = "   (not in the document)" if kind == "out" else ""
        print(f"\nQ{n}. {q}{note}")
        for i in range(min(TOP_SHOW, len(texts))):
            mark = "✓" if rank == i + 1 else " "
            print(f"   #{i + 1}  {scores[i]:.3f} {mark} {short(texts[i], 100)}")
        if rank and rank > TOP_SHOW:
            print(f"   ... correct answer is #{rank} ({scores[rank - 1]:.3f}): {short(texts[rank - 1], 100)}")

    # summary
    ins = [r for r in results if r[2] == "in"]
    outs = [r for r in results if r[2] == "out"]
    hit1 = sum(1 for r in ins if r[3] == 1)
    hit3 = sum(1 for r in ins if r[3] is not None and r[3] <= 3)
    print("\n=== summary ===")
    print(f"hit@1: {hit1}/{len(ins)}   hit@3: {hit3}/{len(ins)}")
    if ins and outs:
        low_in = min(r[4][0] for r in ins)
        high_out = max(r[4][0] for r in outs)
        gap = low_in - high_out
        print(f"lowest top-1 score of real questions:      {low_in:.3f}")
        print(f"highest top-1 score of unrelated questions: {high_out:.3f}")
        print(f"gap: {gap:+.3f}  ->", end=" ")
        if gap >= 0.10:
            print("a score threshold can separate them")
        elif gap > 0:
            print("tiny gap: a threshold alone is risky")
        else:
            print("overlap: a threshold alone CANNOT separate them")


if __name__ == "__main__":
    main()