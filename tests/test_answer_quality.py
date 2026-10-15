
import re


def extract_citations(answer):
    """Extract citation IDs written in square brackets."""
    return set(
        re.findall(r"\[([A-Za-z0-9_-]+)\]", answer)
    )


def check_citations(answer, supplied_ids):
    cited_ids = extract_citations(answer)

    return {
        "has_citations": bool(cited_ids),
        "unsupported_citations": cited_ids - supplied_ids,
        "valid": bool(cited_ids) and cited_ids.issubset(supplied_ids),
    }


def check_required_terms(answer, required_terms):
    """Check whether expected evidence details appear in an answer."""
    normalized = answer.lower()

    return {
        term: term.lower() in normalized
        for term in required_terms
    }


def main():
    # Example output from the current Qwen run.
    # Replace this with a saved generated answer when evaluating
    # future runs.
    answer = """
    In [SDR_2019_00000087], a main battery charger failure was
    reported for a BOEING 777224 aircraft. The charger was
    replaced and an operational check was reported.
    In [SDR_2019_00000000], a main battery charger status message
    cleared after battery assembly replacement and operational
    checks. The underlying cause is not established.
    """

    supplied_ids = {
        "SDR_2019_00000087",
        "SDR_2019_00000000",
    }

    print("=" * 70)
    print("ANSWER QUALITY CHECKS")
    print("=" * 70)

    citation_result = check_citations(answer, supplied_ids)

    print("\nCitation validation:")
    print(f"Has citations: {citation_result['has_citations']}")
    print(
        "Unsupported citations: "
        f"{sorted(citation_result['unsupported_citations'])}"
    )
    print(f"Valid: {citation_result['valid']}")

    required_terms = [
        "SDR_2019_00000087",
        "SDR_2019_00000000",
        "underlying cause is not established",
    ]

    term_results = check_required_terms(answer, required_terms)

    print("\nEvidence coverage:")
    for term, found in term_results.items():
        print(f"{term}: {'PASS' if found else 'FAIL'}")

    print(
        "\nNote: These are structural checks. They cannot prove "
        "that a generated statement accurately represents its source."
    )


if __name__ == "__main__":
    main()
