"""PII redaction engine using Presidio and spaCy en_core_web_sm.

Restricts entity detection to sensitive types, adds custom recognizer for
EMPLOYEE_ID, avoids false positives on locations/dates,
and provides configurable fail-closed / fail-open behavior.
"""

import logging
import re
from typing import List, Optional, Tuple

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .config import Settings, settings

# Default project codenames to redact (configurable via PROJECT_CODENAMES env var)
DEFAULT_PROJECT_CODENAMES = ["Phoenix", "Titan", "Aurora", "Nebula", "Valkyrie"]

logger = logging.getLogger(__name__)

# Strict list of entities to redact per Section 5.2
# DO NOT include LOCATION or DATE_TIME to avoid breaking questions like "What is the capital of Germany?"
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "EMPLOYEE_ID",
    "PROJECT_CODENAME",
]


class PIIRedactionError(Exception):
    """Raised when PII redaction fails in fail-closed mode."""
    pass


class PIIRedactor:
    """PII Redaction engine with Presidio and custom enterprise recognizers."""

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or settings
        self.fail_mode = self.config.PII_FAIL_MODE
        self.supported_entities = list(DEFAULT_ENTITIES)
        self.analyzer: Optional[AnalyzerEngine] = None
        self.anonymizer: Optional[AnonymizerEngine] = None
        self.operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in self.supported_entities
        }
        self._init_presidio()

    def _init_presidio(self) -> None:
        """Initialize Presidio AnalyzerEngine with spaCy en_core_web_sm and custom recognizers."""
        try:
            logger.info("Initializing Presidio AnalyzerEngine with spaCy en_core_web_sm...")
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()

            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.anonymizer = AnonymizerEngine()

            # Custom Recognizer 1: EMPLOYEE_ID (e.g., EMP-12345, EMP-987654)
            emp_pattern = Pattern(
                name="employee_id_pattern",
                regex=r"\bEMP-\d{5,7}\b",
                score=0.85,
            )
            emp_recognizer = PatternRecognizer(
                supported_entity="EMPLOYEE_ID",
                patterns=[emp_pattern],
                context=["employee", "emp", "id", "staff", "badge", "worker"],
            )
            self.analyzer.registry.add_recognizer(emp_recognizer)

            # Custom Recognizer 2: Robust Phone Pattern (catches synthetic/international numbers)
            phone_pattern = Pattern(
                name="robust_phone_pattern",
                regex=r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                score=0.85,
            )
            phone_recognizer = PatternRecognizer(
                supported_entity="PHONE_NUMBER",
                patterns=[phone_pattern],
                context=["phone", "call", "mobile", "cell", "tel", "contact"],
            )
            self.analyzer.registry.add_recognizer(phone_recognizer)

            # Custom Recognizer 3: PROJECT_CODENAME deny-list (configurable via env)
            codenames = getattr(self.config, "PROJECT_CODENAMES", None)
            if codenames:
                codename_list = [c.strip() for c in codenames.split(",") if c.strip()]
            else:
                codename_list = list(DEFAULT_PROJECT_CODENAMES)

            if codename_list:
                # Build regex pattern from deny-list: \b(Phoenix|Titan|Aurora|...)\b
                escaped = [re.escape(name) for name in codename_list]
                codename_pattern = Pattern(
                    name="project_codename_pattern",
                    regex=r"\b(" + "|".join(escaped) + r")\b",
                    score=0.85,
                )
                codename_recognizer = PatternRecognizer(
                    supported_entity="PROJECT_CODENAME",
                    patterns=[codename_pattern],
                    context=["project", "codename", "program", "initiative", "operation"],
                )
                self.analyzer.registry.add_recognizer(codename_recognizer)
                logger.info("Registered PROJECT_CODENAME deny-list recognizer with %d codenames.", len(codename_list))

            logger.info("PIIRedactor initialized successfully with %d entities.", len(self.supported_entities))
        except Exception as e:
            logger.error("Failed to initialize Presidio PII engine: %s", str(e), exc_info=True)
            if self.fail_mode == "closed":
                raise RuntimeError(f"Presidio PII analyzer failed to initialize (fail-mode: closed): {e}") from e
            else:
                logger.warning("Operating in fail-open mode; PII redaction is disabled.")

    def redact(self, text: str) -> Tuple[str, int]:
        """Redact PII from the given text.

        WHY: In an enterprise setting, sensitive user data (names, emails, IDs) must
        never reach the model or logs. We scan inbound prompts and replace entities with
        typed placeholders (e.g., <EMAIL_ADDRESS>) while preserving sentence structure.

        Args:
            text: Input string.

        Returns:
            Tuple of (redacted_text, redaction_count).
        """
        if not text:
            return text, 0

        if self.analyzer is None or self.anonymizer is None:
            if self.fail_mode == "closed":
                raise PIIRedactionError("PII analyzer is not initialized (fail-mode: closed).")
            return text, 0

        try:
            results = self.analyzer.analyze(
                text=text,
                entities=self.supported_entities,
                language="en",
            )

            if not results:
                return text, 0

            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=self.operators,
            )

            redaction_count = len(anonymized.items)
            return anonymized.text, redaction_count
        except Exception as e:
            logger.error("Error during PII redaction: %s", str(e), exc_info=True)
            if self.fail_mode == "closed":
                raise PIIRedactionError(f"PII redaction failed (fail-mode: closed): {e}") from e
            logger.warning("PII redaction failed; continuing with raw text in fail-open mode.")
            return text, 0


# Module-level default instance
_redactor: Optional[PIIRedactor] = None


def get_redactor(cfg: Optional[Settings] = None) -> PIIRedactor:
    """Return or initialize the singleton PIIRedactor."""
    global _redactor
    if _redactor is None or cfg is not None:
        _redactor = PIIRedactor(cfg or settings)
    return _redactor


def redact_pii(text: str) -> Tuple[str, int]:
    """Convenience function to redact PII using the default redactor."""
    return get_redactor().redact(text)