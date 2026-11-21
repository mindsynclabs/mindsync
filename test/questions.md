# RAG test kit — questions and expected answers

Load order: 01_people.txt, 02_zentora_facts.txt, 03_zentora_handbook.pdf
(All facts are invented, so the LLM cannot answer from its own knowledge.
A correct answer proves retrieval + generation worked.)

## A. Single-chunk lookups (retrieval basics)
| Question | Expected |
|---|---|
| Is An male or female? | male |
| What is Cuong's favorite programming language? | Rust |
| What is the name of Hoa's cat? | Mochi |
| Who founded Zentora Labs? | Linh Tran |
| What is Zentora's main product? | QuillBase |
| When was QuillBase 3.2 released? | 14 August 2026 |

## B. PDF content (tests pypdf / OCR path)
| Question | Expected |
|---|---|
| How many days of annual leave do employees get? | 18 |
| How many leave days can be carried over? | up to 5 |
| How many remote days are allowed per week? | 2 |
| How long must passwords be? | at least 14 characters |
| Who is the security officer? | Hoa |
| What is the internet allowance for remote workers? | 500,000 VND |

## C. Reasoning / inference (tests the LLM, not just retrieval)
| Question | Expected |
|---|---|
| Is Binh male? | No, Binh is female |
| Is An female? | No, An is male |
| Who owns a cat and also handles security incidents? | Hoa |
| How many offices does the company have, and where? | 3: Da Nang, HCMC, Singapore |
| At what time does the Singapore office close? | 18:00 |

## D. Must say "not found" (tests that it does NOT hallucinate)
| Question | Expected |
|---|---|
| What is Zentora's annual revenue? | Not found |
| What is Linh Tran's phone number? | Not found |
| Who is the president of France? | Not found (or general answer, depending on ANSWER_MODE) |

## E. Cross-document (tests top_k and multi-source)
| Question | Expected |
|---|---|
| Which employee joined in March 2022 and what is her job? | Dung, finance manager |
| Where is the company retreat and which month? | Phu Quoc island, October |
