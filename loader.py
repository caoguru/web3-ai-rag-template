"""
loader.py
Load and clean all Web3 whitepaper PDFs from the docs/ directory.

Supports .pdf files using pypdf for text extraction.
"""

from pathlib import Path
from pypdf import PdfReader

DOCS_DIR = Path("docs")
SUPPORTED_EXTENSIONS = {".pdf"}


def clean_text(raw_text: str) -> str:
    """
    Normalize extracted PDF text by collapsing whitespace and
    removing empty lines.

    Args:
        raw_text: Raw text extracted from a PDF.

    Returns:
        Cleaned plain text.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)


def load_pdf(filepath: Path) -> str:
    """
    Extract all text from a PDF file.

    Args:
        filepath: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    reader = PdfReader(str(filepath))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def load_single_file(filepath: Path) -> dict | None:
    """
    Load and clean a single PDF document.

    Args:
        filepath: Path to the PDF file.

    Returns:
        A dict with 'source' and 'text', or None if unsupported or empty.
    """
    if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None

    try:
        raw_text = load_pdf(filepath)
    except Exception as error:
        print(f"[SKIP] {filepath.name}: {error}")
        return None

    text = clean_text(raw_text)
    if not text:
        print(f"[SKIP] {filepath.name}: no extractable text (scanned PDF?)")
        return None

    return {"source": filepath.stem, "text": text}


def load_all_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """
    Load every supported PDF from the docs/ directory.

    Args:
        docs_dir: Directory containing whitepaper PDFs.

    Returns:
        A list of dicts, each with 'source' and 'text'.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")

    documents = []
    for filepath in sorted(docs_dir.iterdir()):
        if filepath.is_file():
            doc = load_single_file(filepath)
            if doc:
                documents.append(doc)
                print(f"[LOADED] {filepath.name} ({len(doc['text'])} chars)")

    if not documents:
        raise ValueError(f"No supported PDFs found in {docs_dir}")

    return documents


if __name__ == "__main__":
    docs = load_all_documents()
    print(f"\nTotal documents loaded: {len(docs)}")
    for d in docs:
        print(f" - {d['source']}: {len(d['text'])} chars")