"""Tests for PDF and DOCX document parsers."""

import io
from pathlib import Path
import docx
from PIL import Image
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
    p1.insert_text((50, 50), "First page: 5G RAN deployment specifications, massive MIMO beamforming configuration and cell capacity.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Second page: Cloud core UPF network throughput benchmarks and user plane latency measurements.")
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
def scanned_image_pdf_file(tmp_path: Path) -> Path:
    """Create a 1-page PDF containing a raster image and zero digital text."""
    pdf_path = tmp_path / "scanned_image.pdf"
    img = Image.new("RGB", (100, 100), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(0, 0, 100, 100), stream=buf.getvalue())
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


def test_pdf_parser_rejects_empty_or_scanned_pdf(empty_pdf_file: Path, monkeypatch):
    """Empty PDF without OCR text must raise ValueError stating OCR is unavailable or found no text."""
    monkeypatch.setattr("rag_service.parsers.pdf._TESSERACT_AVAILABLE", False)
    with pytest.raises(ValueError) as excinfo:
        extract_pages_from_pdf(empty_pdf_file)
    assert "No extractable text found in PDF" in str(excinfo.value)
    assert "OCR is unavailable or found no text" in str(excinfo.value)


def test_pdf_parser_ocr_fallback_yields_text_when_available(scanned_image_pdf_file: Path, monkeypatch):
    """When OCR is available and page has no text layer, OCR extracts the page text."""
    monkeypatch.setattr("rag_service.parsers.pdf._TESSERACT_AVAILABLE", True)

    import pytesseract

    monkeypatch.setattr(
        pytesseract,
        "image_to_string",
        lambda img: "Extracted scanned content: Beamforming parameters for carrier aggregation.",
    )

    pages = extract_pages_from_pdf(scanned_image_pdf_file)
    assert len(pages) == 1
    assert pages[0][0] == 1
    assert "Beamforming parameters for carrier aggregation" in pages[0][1]


def test_pdf_parser_image_only_raises_error_when_ocr_unavailable(scanned_image_pdf_file: Path, monkeypatch):
    """When OCR is unavailable, an image-only PDF raises ValueError with the exact error message."""
    monkeypatch.setattr("rag_service.parsers.pdf._TESSERACT_AVAILABLE", False)
    with pytest.raises(ValueError) as excinfo:
        extract_pages_from_pdf(scanned_image_pdf_file)
    assert "No extractable text found in PDF" in str(excinfo.value)
    assert "OCR is unavailable or found no text" in str(excinfo.value)


def test_pdf_parser_normal_text_pdf_never_calls_ocr(tiny_pdf_file: Path, monkeypatch):
    """A PDF with rich digital text must extract text directly without calling OCR."""
    monkeypatch.setattr("rag_service.parsers.pdf._TESSERACT_AVAILABLE", True)

    ocr_called = False

    def mock_image_to_string(img):
        nonlocal ocr_called
        ocr_called = True
        return "Should not be called"

    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_string", mock_image_to_string)

    pages = extract_pages_from_pdf(tiny_pdf_file)
    assert len(pages) == 2
    assert "beamforming" in pages[0][1]
    assert "UPF" in pages[1][1]
    assert ocr_called is False


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
    """parse_document must reject unsupported extensions like .csv with ValueError."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\nval1,val2")
    with pytest.raises(ValueError) as excinfo:
        parse_document(csv_file)
    assert "Unsupported file format" in str(excinfo.value)


def test_parse_document_txt_and_md_formats(tmp_path: Path):
    """parse_document must parse .txt and .md files into page tuples."""
    # 1. Plain text file
    txt_file = tmp_path / "guide.txt"
    txt_file.write_text("This is an operations runbook for incident response.")
    pages_txt = parse_document(txt_file)
    assert len(pages_txt) == 1
    assert pages_txt[0][0] == 1
    assert "incident response" in pages_txt[0][1]

    # 2. Markdown file
    md_file = tmp_path / "readme.md"
    md_file.write_text("# Project Architecture\n\nThe gateway manages Presidio PII and BGE routing.")
    pages_md = parse_document(md_file)
    assert len(pages_md) == 1
    assert "Project Architecture" in pages_md[0][1]

    # 3. Empty text file raises ValueError
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("   \n\t  ")
    with pytest.raises(ValueError) as excinfo:
        parse_document(empty_txt)
    assert "No extractable text found" in str(excinfo.value)



def test_pdf_parser_table_structure_and_chunking(tmp_path: Path):
    """Fast test verifying PDF table extraction serializes rows with headers and chunking keeps rows intact."""
    from rag_service.chunking.character import split_character_text
    from rag_service.chunking.structure import split_structure_text

    pdf_path = tmp_path / "table_test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 40), "Configuration parameters for the alerting service runtime environment.")
    page.draw_rect(pymupdf.Rect(50, 50, 350, 110), color=(0, 0, 0), width=1)
    page.draw_line(pymupdf.Point(50, 80), pymupdf.Point(350, 80), color=(0, 0, 0), width=1)
    page.draw_line(pymupdf.Point(200, 50), pymupdf.Point(200, 110), color=(0, 0, 0), width=1)
    page.insert_text((60, 70), "Setting")
    page.insert_text((210, 70), "Value")
    page.insert_text((60, 100), "RETENTION_DAYS")
    page.insert_text((210, 100), "45")
    doc.save(str(pdf_path))
    doc.close()

    pages = extract_pages_from_pdf(pdf_path)
    assert len(pages) == 1
    page_text = pages[0][1]
    assert "Setting: RETENTION_DAYS | Value: 45" in page_text

    # Verify character chunking keeps the table row unbroken
    char_chunks = split_character_text(page_text, chunk_size=500, overlap=50)
    assert any("Setting: RETENTION_DAYS | Value: 45" in c for c in char_chunks)

    # Verify structure chunking keeps the table row unbroken
    struct_chunks = split_structure_text(page_text, max_chunk_size=1000)
    assert any("Setting: RETENTION_DAYS | Value: 45" in c for c in struct_chunks)

