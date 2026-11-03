from fastapi import UploadFile
import os

class DocumentLoader:
    """Handles loading and text extraction from various uploaded file formats."""

    @staticmethod
    async def extract_text(file: UploadFile) -> str:
        """Read content from an uploaded file based on its extension."""
        filename = file.filename.lower()
        content_bytes = await file.read()
        
        # Handle plain text and markdown files
        if filename.endswith((".txt", ".md", ".csv")):
            try:
                return content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Fallback to latin-1 if utf-8 decoding fails
                return content_bytes.decode("latin-1")
                
        # Add future PDF or docx extractors here if needed
        elif filename.endswith(".pdf"):
            raise NotImplementedError("PDF parsing requires an extra library like pypdf. Use text/markdown for now.")
            
        else:
            # Default attempt to decode as text
            return content_bytes.decode("utf-8", errors="ignore")

document_loader = DocumentLoader()