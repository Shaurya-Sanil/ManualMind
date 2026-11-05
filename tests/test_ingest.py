"""
Tests for Module M1: Document Ingestion & Chunking
"""

import pytest
from pathlib import Path

from nlp.ingest import (
    ingest_manual, chunk_text_content, is_heading_candidate,
    ScannedPDFError, ManualChunk, extract_from_txt_or_md
)


def test_is_heading_candidate():
    assert is_heading_candidate("1. Product Overview") == 1
    assert is_heading_candidate("1.1 Package Contents") == 2
    assert is_heading_candidate("1.2.3 Hardware Subsystem") == 3
    assert is_heading_candidate("Chapter 2: Quick Start") == 1
    assert is_heading_candidate("Normal body paragraph without numbers", font_size=10.0, avg_font_size=10.0) is None
    assert is_heading_candidate("Large Heading Text", font_size=18.0, avg_font_size=10.0) == 1


def test_chunk_text_content_boundaries():
    sample_text = (
        "Paragraph one introduces the router and explains the wireless standards. " * 8 + "\n\n" +
        "Step 1: Connect the power adapter to the wall outlet.\n\n" +
        "Step 2: Connect the blue WAN cable from the modem to the router.\n\n" +
        "Step 3: Verify the POWER LED turns solid white.\n\n" +
        "Paragraph three covers additional troubleshooting tips. " * 8
    )

    chunks = chunk_text_content(
        text=sample_text,
        manual_name="test_manual.pdf",
        page=1,
        section_path="Hardware Installation > Quick Start",
        chunk_index_start=1,
        target_words=50,
        overlap_words=15
    )

    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, ManualChunk)
        assert c.manual == "test_manual.pdf"
        assert c.page == 1
        assert "Hardware Installation" in c.section_path
        assert c.word_count > 0


def test_chunk_preserves_numbered_steps():
    text_with_steps = (
        "Follow these steps to reset the router:\n\n"
        "1. Power on the router.\n\n"
        "2. Press and hold the reset pin for 10 seconds.\n\n"
        "3. Wait for the LEDs to blink amber."
    )
    chunks = chunk_text_content(
        text=text_with_steps,
        manual_name="test.pdf",
        page=2,
        section_path="Reset",
        chunk_index_start=1
    )
    assert len(chunks) == 1
    assert chunks[0].has_numbered_steps is True


def test_extract_from_txt_and_md(tmp_path: Path):
    md_file = tmp_path / "sample.md"
    md_file.write_text("# 1. Introduction\nThis is a sample router guide.\n\n## 1.1 Setup\nStep 1: Plug in power.", encoding="utf-8")
    
    pages = extract_from_txt_or_md(md_file)
    assert len(pages) >= 1
    assert len(pages[0]["blocks"]) >= 2


def test_empty_document_raises_error(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_from_txt_or_md(empty_file)


def test_ingest_aurora_manual_pdf():
    pdf_path = Path("data/manuals/aurora_ax3000_manual.pdf")
    if not pdf_path.exists():
        pytest.skip("Aurora manual PDF not generated yet.")

    doc = ingest_manual(pdf_path)
    assert doc.manual_name == "aurora_ax3000_manual.pdf"
    assert doc.total_pages >= 15
    assert len(doc.sections) >= 5
    assert len(doc.chunks) >= 15
    assert doc.vocabulary_size > 100
