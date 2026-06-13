import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("oprel.utils.file_parser")

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to parse PDF with pypdf {file_path}: {e}")
        text = ""

    # Fallback to pdfplumber if pypdf failed or returned no text
    if not text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Failed to parse PDF with pdfplumber {file_path}: {e}")
            
    return text.strip()

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        import docx
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text).strip()
    except ImportError:
        logger.error("python-docx not installed. Cannot parse DOCX.")
        return "[Error: python-docx not installed]"
    except Exception as e:
        logger.error(f"Failed to parse DOCX {file_path}: {e}")
        return f"[Error parsing DOCX: {e}]"

def extract_text(file_path: Path) -> str:
    """
    General purpose text extractor for various file types.
    Supports .txt, .md, .pdf, .docx, and common source code files.
    """
    suffix = file_path.suffix.lower()
    
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix == ".docx":
        return extract_text_from_docx(file_path)
    elif suffix in [".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".yaml", ".yml"]:
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return f"[Error reading file: {e}]"
    else:
        # Fallback to simple read for unknown types
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""
