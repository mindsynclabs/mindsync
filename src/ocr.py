"""PDF text extraction with a vision-model OCR fallback (qwen2.5vl via Ollama)."""
import io
import logging
import os

import pymupdf  # was: import fitz # PyMuPDF: renders PDF pages to images
import ollama
from pypdf import PdfReader

log = logging.getLogger("ocr")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OCR_MODEL = os.getenv("OCR_MODEL", "qwen2.5vl:7b")
# auto   = OCR only pages with no/little extractable text (scanned pages)
# always = OCR every page
# off    = never OCR (text-based PDFs only)
PDF_OCR = os.getenv("PDF_OCR", "auto")
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "30"))
# Safety cap: 7B vision model on CPU is slow, so limit pages per upload
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "10"))
OCR_DPI = int(os.getenv("OCR_DPI", "150"))

OCR_PROMPT = (
    "Transcribe all the text in this page image exactly as written, "
    "in natural reading order. Output only the transcribed text, with no "
    "commentary. Use markdown tables for tables."
)

_client = ollama.Client(host=OLLAMA_URL, timeout=900)


def ocr_page(png_bytes: bytes) -> str:
    try:
        resp = _client.chat(
            model=OCR_MODEL,
            messages=[{"role": "user", "content": OCR_PROMPT, "images": [png_bytes]}],
            options={"temperature": 0},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(
            f"OCR failed with model '{OCR_MODEL}' "
            f"(is it pulled? try: ollama pull {OCR_MODEL}): {e}"
        )


def extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    texts = [(p.extract_text() or "").strip() for p in reader.pages]

    if PDF_OCR == "off":
        return "\n".join(t for t in texts if t)

    needs_ocr = [
        i for i, t in enumerate(texts) if PDF_OCR == "always" or len(t) < OCR_MIN_CHARS
    ]
    if len(needs_ocr) > OCR_MAX_PAGES:
        log.warning(
            "%d pages need OCR, only doing the first %d (OCR_MAX_PAGES)",
            len(needs_ocr),
            OCR_MAX_PAGES,
        )
        needs_ocr = needs_ocr[:OCR_MAX_PAGES]

    if needs_ocr:
        doc = pymupdf.open(stream=content, filetype="pdf")
        try:
            for n, i in enumerate(needs_ocr, 1):
                log.info("OCR page %d (%d/%d)", i + 1, n, len(needs_ocr))
                png = doc[i].get_pixmap(dpi=OCR_DPI).tobytes("png")
                texts[i] = ocr_page(png)
        finally:
            doc.close()

    return "\n".join(t for t in texts if t)
