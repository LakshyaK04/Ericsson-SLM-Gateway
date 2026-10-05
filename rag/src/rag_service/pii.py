"""PII redaction helper for RAG document ingestion.

Provides lightweight regex-based masking for documents during ingestion
when PII_REDACTION_ON_INGEST=true, without requiring heavy external dependencies.
"""

import re
from typing import List, Optional, Tuple

DEFAULT_CODENAMES = ["Phoenix", "Titan", "Aurora", "Nebula", "Valkyrie"]

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
EMP_ID_REGEX = re.compile(r"\bEMP-\d{5,7}\b")
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")


def redact_ingest_text(text: str, codenames: Optional[List[str]] = None) -> Tuple[str, int]:
    """Redact PII from ingested document text.

    Returns:
        Tuple of (sanitized_text, redaction_count).
    """
    if not text:
        return text, 0

    count = 0

    def _replace_email(match):
        nonlocal count
        count += 1
        return "<EMAIL_ADDRESS>"

    def _replace_phone(match):
        nonlocal count
        count += 1
        return "<PHONE_NUMBER>"

    def _replace_emp(match):
        nonlocal count
        count += 1
        return "<EMPLOYEE_ID>"

    def _replace_ip(match):
        nonlocal count
        count += 1
        return "<IP_ADDRESS>"

    def _replace_cc(match):
        nonlocal count
        count += 1
        return "<CREDIT_CARD>"

    def _replace_codename(match):
        nonlocal count
        count += 1
        return "<PROJECT_CODENAME>"

    redacted = EMAIL_REGEX.sub(_replace_email, text)
    redacted = PHONE_REGEX.sub(_replace_phone, redacted)
    redacted = EMP_ID_REGEX.sub(_replace_emp, redacted)
    redacted = IP_REGEX.sub(_replace_ip, redacted)
    redacted = CREDIT_CARD_REGEX.sub(_replace_cc, redacted)

    names = codenames or DEFAULT_CODENAMES
    if names:
        escaped = [re.escape(name) for name in names if name]
        if escaped:
            codename_pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b")
            redacted = codename_pattern.sub(_replace_codename, redacted)

    return redacted, count
