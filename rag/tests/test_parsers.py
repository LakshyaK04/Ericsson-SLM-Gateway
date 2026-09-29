"""Tests for PDF and DOCX document parsers."""

from pathlib import Path
import docx
import pymupdf
import pytest

from rag_service.parsers import parse_document
from rag_service.parsers.docx import extract_pages_from_docx
from rag_service.parsers.pdf import extract_pages_from_pdf


@pytest.fixture
def tiny_pdf_file(tmp_path: Path) -> Path:
    """Create a minimal valid 2-page PDF fixture."""
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "First page: 5G RAN deployment specifications and beamforming.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Second page: Cloud core UPF network throughput benchmarks.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf_file(tmp_path: Path) -> Path:
    """Create an empty 1-page PDF fixture with zero text."""
    pdf_path = tmp_path / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()  # blank page
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def tiny_docx_file(tmp_path: Path) -> Path:
    """Create a minimal valid DOCX fixture with headings and tables."""
    docx_path = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_heading("Network Architecture", level=1)
    doc.add_paragraph("This document details the Standalone 5G core configuration.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Parameter"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "MaxThroughput"
    table.cell(1, 1).text = "10Gbps"
    doc.save(str(docx_path))
    return docx_path


def test_pdf_parser_extracts_page_numbers_and_text(tiny_pdf_file: Path):
    """PDF parser must extract 1-indexed page tuples with clean text."""
    pages = extract_pages_from_pdf(tiny_pdf_file)
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "beamforming" in pages[0][1]
    assert pages[1][0] == 2
    assert "UPF" in pages[1][1]


def test_pdf_parser_rejects_empty_or_scanned_pdf(empty_pdf_file: Path):
    """Empty PDF must raise ValueError stating OCR is not supported."""
    with pytest.raises(ValueError) as excinfo:
        extract_pages_from_pdf(empty_pdf_file)
    assert "OCR is not supported" in str(excinfo.value)


def test_docx_parser_extracts_headings_and_tables(tiny_docx_file: Path):
    """DOCX parser must extract paragraphs, headings, and table cells."""
    pages = extract_pages_from_docx(tiny_docx_file)
    assert len(pages) == 1
    page_num, text = pages[0]
    assert page_num == 1
    assert "Network Architecture" in text
    assert "Standalone 5G" in text
    assert "MaxThroughput" in text
    assert "10Gbps" in text


def test_parse_document_dispatcher_rejects_unsupported_extensions(tmp_path: Path):
    """parse_document must reject non-pdf/docx extensions with ValueError."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Plain text notes.")
    with pytest.raises(ValueError) as excinfo:
        parse_document(txt_file)
    assert "Unsupported file format" in str(excinfo.value)
