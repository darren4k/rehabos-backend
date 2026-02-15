"""Document text extraction for referral PDFs and faxed images."""

import logging

logger = logging.getLogger(__name__)

SUPPORTED_PDF_TYPES = {"application/pdf"}
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/tiff"}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file.

    Handles both text-based and scanned PDFs. For scanned PDFs with no
    extractable text, returns a notice that OCR would be needed.

    Args:
        file_bytes: Raw PDF file bytes.

    Returns:
        Extracted text content.
    """
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise ImportError(
            "pymupdf is required for PDF extraction. Install with: pip install pymupdf"
        ) from e

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())

    doc.close()

    if not pages:
        logger.warning("No text extracted from PDF — may be a scanned document requiring OCR")
        return "[No extractable text found — scanned document may require OCR]"

    return "\n\n".join(pages)


def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from a faxed image.

    Currently returns a placeholder. In production, this would use an LLM
    with vision capabilities or a dedicated OCR service.

    Args:
        file_bytes: Raw image file bytes.

    Returns:
        Extracted text or placeholder.
    """
    # TODO: Implement vision-based OCR using LLM with image input
    # Would encode to base64 and send to a vision model
    logger.info("Image OCR requested — vision-based extraction not yet implemented")
    return "[Vision OCR placeholder — image text extraction would go here]"


def extract_text(file_bytes: bytes, content_type: str) -> str:
    """Route to the appropriate text extractor based on MIME type.

    Args:
        file_bytes: Raw file bytes.
        content_type: MIME type of the file.

    Returns:
        Extracted text content.

    Raises:
        ValueError: If the content type is not supported.
    """
    content_type = content_type.lower().strip()

    if content_type in SUPPORTED_PDF_TYPES:
        return extract_text_from_pdf(file_bytes)
    elif content_type in SUPPORTED_IMAGE_TYPES:
        return extract_text_from_image(file_bytes)
    else:
        raise ValueError(
            f"Unsupported content type: {content_type}. "
            f"Supported: {SUPPORTED_PDF_TYPES | SUPPORTED_IMAGE_TYPES}"
        )
