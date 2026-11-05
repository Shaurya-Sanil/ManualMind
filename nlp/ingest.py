"""
ManualMind Ingestion Module (M1)
Extracts structured text from PDF, TXT, and Markdown documents.
Detects heading hierarchy, preserves list items (numbered steps, bullets),
splits content into semantic overlapping chunks (120-200 words),
and tracks complete metadata (manual, page, section path, chunk id).
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ManualChunk(BaseModel):
    chunk_id: str
    manual: str
    page: int
    section_path: str
    content: str
    word_count: int
    has_numbered_steps: bool = False
    has_bullets: bool = False


class IngestedDocument(BaseModel):
    manual_name: str
    source_path: str
    total_pages: int
    sections: List[str]
    chunks: List[ManualChunk]
    raw_text: str
    vocabulary_size: int


class ScannedPDFError(ValueError):
    """Raised when a PDF contains no extractable text layer (scanned)."""
    pass


IMPERATIVE_VERBS = {
    "connect", "plug", "press", "select", "click", "slide", "fold", "turn", "insert",
    "push", "open", "close", "switch", "download", "remove", "place", "align", "attach",
    "hold", "release", "repeat", "ensure", "verify", "check", "rotate", "wait", "power",
    "charge", "tap", "swipe", "enter", "type", "navigate", "go", "choose", "install",
    "adjust", "disconnect", "unplug", "locate", "set", "use", "follow", "keep", "make",
    "put", "take", "pull", "read", "refer", "see", "allow", "do", "restart", "reboot"
}


def is_heading_candidate(line: str, font_size: float = 0.0, avg_font_size: float = 10.0) -> Optional[int]:
    """
    Determines if a text line is a heading based on numbering patterns and font size.
    Returns the heading level (1, 2, or 3) if candidate, else None.
    Distinguishes section headings from numbered procedural steps.
    """
    clean = line.strip()
    if not clean:
        return None

    # Filter running headers/footers
    if re.match(r"^\s*\d+\s*(?:\||\u2502|\/)\s*[A-Z\s]{2,}\s*$", clean):
        return None
    if re.match(r"^\s*[A-Z\s]{2,}\s*(?:\||\u2502|\/)\s*\d+\s*$", clean):
        return None
    if re.match(r"^\s*\d+\s*$", clean):
        return None

    # Explicit chapter/appendix keywords
    m_chap = re.match(r"^(Chapter|Section|Appendix)\s+\d+", clean, re.IGNORECASE)
    if m_chap:
        return 1

    # Numbered heading patterns: e.g. "1. Product Overview" or "1.1 Package Contents"
    m_num = re.match(r"^(\d+(?:\.\d+)*\.?)\s+([A-Za-z].*)$", clean)
    if m_num:
        num_part = m_num.group(1).rstrip(".")
        rest = m_num.group(2).strip()
        words = rest.split()
        first_word = words[0].lower().rstrip(".,:;") if words else ""

        # Single number (e.g. "1." or "2."): Disambiguate procedural step vs section heading
        if "." not in num_part:
            # Procedural action verb -> step, not heading
            if first_word in IMPERATIVE_VERBS:
                return None
            # Ends in sentence-ending punctuation with more than 4 words -> step/sentence
            if clean.endswith((".", ";")) and len(words) > 4:
                return None
            # Normal body font size -> not a heading
            if font_size > 0 and avg_font_size > 0 and font_size < avg_font_size * 1.15:
                return None
            return 1
        else:
            # Hierarchical numbering like "1.1", "1.2.3"
            dots = num_part.count(".")
            return min(3, dots + 1)

    # Font size based heuristic (at least 20% larger than average font size)
    if font_size >= avg_font_size * 1.20 and len(clean) < 80:
        if font_size >= avg_font_size * 1.45:
            return 1
        return 2

    return None


def extract_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts structured pages and blocks from a PDF using PyMuPDF (fitz).
    Preserves page numbers, block structures, font sizes, and detects headings.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required for PDF ingestion. Please install via pip install pymupdf.")

    doc = fitz.open(str(pdf_path))
    pages_data = []
    total_chars = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_index = page_num + 1  # 1-based page numbering
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
        
        extracted_blocks = []
        for b in blocks:
            if b.get("type") == 0:  # Text block
                block_lines = []
                max_font_size = 0.0
                for line in b.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        text = span.get("text", "").replace("\xad", "")
                        line_text += text
                        size = span.get("size", 10.0)
                        if size > max_font_size:
                            max_font_size = size
                    clean_l = line_text.strip()
                    if clean_l:
                        # Filter running headers/footers (e.g. '2 8 | E N G' with wide tracking, or standalone page numbers)
                        if re.match(r"^\s*[\d\s]+\s*(?:\||\u2502|\/|\-)\s*[A-Za-z\s]+\s*$", clean_l) or \
                           re.match(r"^\s*[A-Za-z\s]+\s*(?:\||\u2502|\/|\-)\s*[\d\s]+\s*$", clean_l) or \
                           re.match(r"^\s*\d+\s*$", clean_l):
                            continue
                        block_lines.append(clean_l)
                
                block_full_text = "\n".join(block_lines)
                total_chars += len(block_full_text)
                if block_full_text.strip():
                    extracted_blocks.append({
                        "text": block_full_text,
                        "lines": block_lines,
                        "font_size": max_font_size,
                        "bbox": b.get("bbox", [])
                    })

        pages_data.append({
            "page": page_index,
            "blocks": extracted_blocks
        })

    doc.close()

    if total_chars < 50:
        raise ScannedPDFError(
            f"The PDF file '{pdf_path.name}' contains no extractable text layer (scanned or image-only). "
            "ManualMind strictly operates on text-based PDFs; OCR is not attempted."
        )

    return pages_data


def extract_from_txt_or_md(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts structured text from a Markdown (.md) or Plain Text (.txt) file.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise ValueError(f"The document '{file_path.name}' is empty.")

    lines = text.splitlines()
    pages_data = []
    current_page_blocks = []
    line_count = 0
    page_num = 1

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        font_size = 10.0
        # Markdown heading detection
        if line_clean.startswith("#"):
            hashes = len(line_clean) - len(line_clean.lstrip("#"))
            font_size = 16.0 if hashes == 1 else (13.0 if hashes == 2 else 11.0)
            line_clean = line_clean.lstrip("#").strip()

        current_page_blocks.append({
            "text": line_clean,
            "lines": [line_clean],
            "font_size": font_size
        })
        line_count += 1

        # Simulate ~40 lines per page for TXT/MD files
        if line_count >= 40:
            pages_data.append({
                "page": page_num,
                "blocks": current_page_blocks
            })
            current_page_blocks = []
            line_count = 0
            page_num += 1

    if current_page_blocks:
        pages_data.append({
            "page": page_num,
            "blocks": current_page_blocks
        })

    return pages_data


def chunk_text_content(
    text: str,
    manual_name: str,
    page: int,
    section_path: str,
    chunk_index_start: int,
    target_words: int = 150,
    overlap_words: int = 30
) -> List[ManualChunk]:
    """
    Splits text into chunks of roughly 120-200 words with overlap,
    preserving bullet points and numbered steps.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: List[ManualChunk] = []
    current_paras: List[str] = []
    current_word_count = 0
    chunk_idx = chunk_index_start

    step_regex = re.compile(r"(?:^|[\r\n]+)\s*(?:Step\s+\d+:|\d+[\.\)])\s+[A-Za-z]", re.IGNORECASE)
    bullet_regex = re.compile(r"(?:^|[\r\n]+)\s*[•\*\-\+]\s+[A-Za-z]")

    for para in paragraphs:
        p_words = len(para.split())
        if p_words == 0:
            continue

        if current_word_count + p_words > (target_words + overlap_words) and current_word_count >= 80:
            content_str = "\n\n".join(current_paras)
            has_steps = bool(step_regex.search(content_str))
            has_bullets = bool(bullet_regex.search(content_str))

            chunk_id = f"{Path(manual_name).stem}_chunk_{chunk_idx:04d}"
            chunks.append(ManualChunk(
                chunk_id=chunk_id,
                manual=manual_name,
                page=page,
                section_path=section_path,
                content=content_str,
                word_count=current_word_count,
                has_numbered_steps=has_steps,
                has_bullets=has_bullets
            ))
            chunk_idx += 1
            # Retain overlap paragraph if available
            current_paras = [current_paras[-1], para] if len(current_paras) > 1 else [para]
            current_word_count = sum(len(p.split()) for p in current_paras)
        else:
            current_paras.append(para)
            current_word_count += p_words

    if current_paras:
        content_str = "\n\n".join(current_paras)
        has_steps = bool(step_regex.search(content_str))
        has_bullets = bool(bullet_regex.search(content_str))
        chunk_id = f"{Path(manual_name).stem}_chunk_{chunk_idx:04d}"
        chunks.append(ManualChunk(
            chunk_id=chunk_id,
            manual=manual_name,
            page=page,
            section_path=section_path,
            content=content_str,
            word_count=current_word_count,
            has_numbered_steps=has_steps,
            has_bullets=has_bullets
        ))

    return chunks


def ingest_manual(file_path: Path) -> IngestedDocument:
    """
    Ingests a PDF, Markdown, or TXT manual, detects section hierarchy,
    generates structured chunks, and gathers document vocabulary statistics.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        pages_data = extract_from_pdf(file_path)
    elif suffix in [".md", ".markdown", ".txt"]:
        pages_data = extract_from_txt_or_md(file_path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats are: .pdf, .md, .txt")

    # Build section hierarchy & collect chunks
    current_hierarchy: List[str] = ["General Information"]
    all_chunks: List[ManualChunk] = []
    all_sections: List[str] = []
    full_text_collector: List[str] = []
    chunk_counter = 1

    # Compute document-wide average font size for reliable heading detection
    all_font_sizes = [
        b["font_size"]
        for page_item in pages_data
        for b in page_item.get("blocks", [])
        if b.get("font_size", 0.0) > 0.0
    ]
    avg_font_size = (sum(all_font_sizes) / len(all_font_sizes)) if all_font_sizes else 10.0

    for page_item in pages_data:
        page_num = page_item["page"]
        blocks = page_item["blocks"]
        
        page_text_segments = []
        for block in blocks:
            text = block["text"]
            font_size = block.get("font_size", 10.0)
            lines = block.get("lines", [text])
            first_line = lines[0] if lines else text

            heading_lvl = is_heading_candidate(first_line, font_size=font_size, avg_font_size=avg_font_size)
            if heading_lvl:
                clean_heading = re.sub(r"^[#\s]+", "", first_line).strip()
                # Adjust hierarchy stack
                if heading_lvl == 1:
                    current_hierarchy = [clean_heading]
                elif heading_lvl == 2:
                    current_hierarchy = [current_hierarchy[0], clean_heading]
                else:
                    if len(current_hierarchy) >= 2:
                        current_hierarchy = [current_hierarchy[0], current_hierarchy[1], clean_heading]
                    else:
                        current_hierarchy.append(clean_heading)

                sec_path = " > ".join(current_hierarchy)
                if sec_path not in all_sections:
                    all_sections.append(sec_path)
                
                # Retain the heading text along with any subsequent lines in the segment
                page_text_segments.append(text)
            else:
                page_text_segments.append(text)

        page_combined_text = "\n\n".join(page_text_segments).strip()
        if page_combined_text:
            full_text_collector.append(page_combined_text)
            sec_path = " > ".join(current_hierarchy)
            chunks = chunk_text_content(
                text=page_combined_text,
                manual_name=file_path.name,
                page=page_num,
                section_path=sec_path,
                chunk_index_start=chunk_counter,
                target_words=150,
                overlap_words=30
            )
            all_chunks.extend(chunks)
            chunk_counter += len(chunks)

    full_raw_text = "\n\n".join(full_text_collector)
    
    # Simple whitespace tokenization to compute vocabulary size
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", full_raw_text.lower())
    vocab_size = len(set(tokens))

    return IngestedDocument(
        manual_name=file_path.name,
        source_path=str(file_path.resolve()),
        total_pages=len(pages_data),
        sections=all_sections,
        chunks=all_chunks,
        raw_text=full_raw_text,
        vocabulary_size=vocab_size
    )


class ManualManager:
    """Singleton-style manager to load, cache, and switch active manuals."""
    def __init__(self, manuals_dir: Path):
        self.manuals_dir = manuals_dir
        self.manuals_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, IngestedDocument] = {}
        self.active_manual_id: Optional[str] = None

    def load_manual(self, filename: str) -> IngestedDocument:
        if filename in self._cache:
            return self._cache[filename]

        file_path = self.manuals_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Manual file not found: {file_path}")

        doc = ingest_manual(file_path)
        self._cache[filename] = doc
        if not self.active_manual_id:
            self.active_manual_id = filename
        return doc

    def get_active_manual(self) -> IngestedDocument:
        if not self.active_manual_id:
            # Look for default manual
            default_manual = self.manuals_dir / "aurora_ax3000_manual.pdf"
            if default_manual.exists():
                return self.load_manual("aurora_ax3000_manual.pdf")
            # If no manual loaded, find any pdf
            all_pdfs = list(self.manuals_dir.glob("*.pdf"))
            if all_pdfs:
                return self.load_manual(all_pdfs[0].name)
            raise FileNotFoundError("No manuals found in storage directory.")
        return self._cache[self.active_manual_id]

    def set_active_manual(self, filename: str) -> IngestedDocument:
        doc = self.load_manual(filename)
        self.active_manual_id = filename
        return doc

    def list_manuals(self) -> List[Dict[str, Any]]:
        results = []
        for f in self.manuals_dir.iterdir():
            if f.suffix.lower() in [".pdf", ".txt", ".md"]:
                # Try reading from cache or file stat
                is_active = (f.name == self.active_manual_id)
                cached = self._cache.get(f.name)
                if cached:
                    results.append({
                        "id": f.name,
                        "name": f.stem.replace("_", " ").title(),
                        "filename": f.name,
                        "total_pages": cached.total_pages,
                        "total_sections": len(cached.sections),
                        "total_chunks": len(cached.chunks),
                        "vocabulary_size": cached.vocabulary_size,
                        "is_active": is_active,
                        "file_size_bytes": f.stat().st_size
                    })
                else:
                    results.append({
                        "id": f.name,
                        "name": f.stem.replace("_", " ").title(),
                        "filename": f.name,
                        "total_pages": 0,
                        "total_sections": 0,
                        "total_chunks": 0,
                        "vocabulary_size": 0,
                        "is_active": is_active,
                        "file_size_bytes": f.stat().st_size
                    })
        return results
