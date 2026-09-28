from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


def redact_pii(text: str) -> str:
    """
    Detect and replace personally identifiable information
    in the input text.
    """

    results = analyzer.analyze(
        text=text,
        language="en",
    )

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    )

    return anonymized.text