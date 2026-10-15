
import re

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.generation.answer_generator import AnswerGenerator


EXPECTED_CHARGER_IDS = {
    "SDR_2019_00000087",
    "SDR_2019_00000000",
}

EXPECTED_FLASHLIGHT_IDS = {
    "SDR_2019_00000060",
}


def filter_records(results, keyword):
    """
    Retain records whose text or selected metadata contains
    the requested keyword.
    """
    keyword = keyword.lower()
    filtered = []

    for result in results:
        metadata = result.get("metadata", {})

        searchable_text = " ".join(
            [
                result.get("text", ""),
                str(metadata.get("component_part_name", "")),
                str(metadata.get("location_on_aircraft", "")),
            ]
        ).lower()

        if keyword in searchable_text:
            filtered.append(result)

    return filtered


def source_ids(results):
    return {
        result.get("source_record_id")
        for result in results
        if result.get("source_record_id")
    }


def evaluate_retrieval(label, results, expected_ids, keyword):
    """
    Report whether expected records were found and calculate
    precision/recall against the small, manually specified
    expected source-ID set.
    """
    filtered = filter_records(results, keyword)
    found_ids = source_ids(filtered)

    true_positives = found_ids & expected_ids
    missing_ids = expected_ids - found_ids

    precision = (
        len(true_positives) / len(found_ids)
        if found_ids
        else 0.0
    )

    recall = (
        len(true_positives) / len(expected_ids)
        if expected_ids
        else 0.0
    )

    print(f"\n{'=' * 70}")
    print(f"RETRIEVAL EVALUATION: {label}")
    print("=" * 70)
    print(f"Candidates retrieved: {len(results)}")
    print(f"Records after filtering: {len(filtered)}")
    print(f"Expected IDs: {sorted(expected_ids)}")
    print(f"Found IDs: {sorted(found_ids)}")
    print(f"Missing expected IDs: {sorted(missing_ids)}")
    print(f"Precision against expected IDs: {precision:.2%}")
    print(f"Recall against expected IDs: {recall:.2%}")

    return filtered, {
        "precision": precision,
        "recall": recall,
        "missing_ids": missing_ids,
    }


def evaluate_citations(answer, supplied_results):
    """
    Check that every source ID cited in the answer was
    included in the supplied evidence.
    """
    cited_ids = set(
        re.findall(r"\[([A-Za-z0-9_-]+)\]", answer)
    )
    supplied_ids = source_ids(supplied_results)

    unsupported_ids = cited_ids - supplied_ids

    print(f"\nCited source IDs: {sorted(cited_ids)}")
    print(f"Supplied source IDs: {sorted(supplied_ids)}")

    if not cited_ids:
        print("Citation check: FAIL - no source IDs cited.")
    elif unsupported_ids:
        print(
            "Citation check: FAIL - citations not supplied "
            f"as evidence: {sorted(unsupported_ids)}"
        )
    else:
        print("Citation check: PASS - all cited IDs were supplied.")

    return not unsupported_ids and bool(cited_ids)


def main():
    print("Loading embedding model...")
    embedder = EmbeddingModel()

    print("Connecting to pilot Qdrant collection...")
    store = VectorStore(
        storage_path="storage/qdrant_pilot",
        collection_name="aviation_pilot",
        vector_size=384,
    )

    generator = AnswerGenerator(model="qwen2.5:3b")

    try:
        # Test 1: Battery charger retrieval
        charger_question = (
            "What does the evidence report about "
            "main aircraft battery charger failures?"
        )

        charger_vector = embedder.embed_query(charger_question)
        charger_candidates = store.search(
            charger_vector,
            limit=10,
        )

        charger_results, charger_metrics = evaluate_retrieval(
            label="Battery charger",
            results=charger_candidates,
            expected_ids=EXPECTED_CHARGER_IDS,
            keyword="charger",
        )

        # Test 2: Flashlight battery retrieval
        flashlight_question = (
            "What do the records report about an aircraft "
            "flashlight battery that is not charging?"
        )

        flashlight_vector = embedder.embed_query(
            flashlight_question
        )
        flashlight_candidates = store.search(
            flashlight_vector,
            limit=10,
        )

        flashlight_results, flashlight_metrics = (
            evaluate_retrieval(
                label="Flashlight battery",
                results=flashlight_candidates,
                expected_ids=EXPECTED_FLASHLIGHT_IDS,
                keyword="flashlight",
            )
        )

        # Test 3: Generate an answer from charger evidence
        print(f"\n{'=' * 70}")
        print("ANSWER GENERATION EVALUATION")
        print("=" * 70)

        answer = generator.generate(
            question=charger_question,
            results=charger_results,
        )

        print("\nGenerated answer:\n")
        print(answer)

        citation_pass = evaluate_citations(
            answer,
            charger_results,
        )

        # Summary
        print(f"\n{'=' * 70}")
        print("EVALUATION SUMMARY")
        print("=" * 70)

        print(
            "Charger expected records found: "
            f"{len(charger_metrics['missing_ids']) == 0}"
        )
        print(
            "Flashlight expected records found: "
            f"{len(flashlight_metrics['missing_ids']) == 0}"
        )
        print(f"Citation validation passed: {citation_pass}")

        print(
            "\nNote: These checks use a small, manually defined "
            "pilot set. They do not establish accuracy across "
            "the complete dataset or prove factual correctness."
        )

    finally:
        store.close()


if __name__ == "__main__":
    main()
