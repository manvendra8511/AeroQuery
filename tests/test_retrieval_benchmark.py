
import json
from pathlib import Path

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


BENCHMARK_PATH = Path("tests/retrieval_benchmark.json")


def main():
    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    embedder = EmbeddingModel()

    store = VectorStore(
        storage_path="storage/qdrant_pilot",
        collection_name="aviation_pilot",
        vector_size=384,
    )

    total_relevant = 0
    total_retrieved_relevant = 0
    reciprocal_ranks = []

    try:
        for case in benchmark:
            query = case["query"]
            relevant_ids = set(case["relevant_source_ids"])

            query_vector = embedder.embed_query(query)
            results = store.search(query_vector, limit=10)

            ranked_ids = [
                result.get("source_record_id")
                for result in results
            ]

            hits = set(ranked_ids) & relevant_ids
            total_relevant += len(relevant_ids)
            total_retrieved_relevant += len(hits)

            reciprocal_rank = 0.0

            for rank, source_id in enumerate(ranked_ids, start=1):
                if source_id in relevant_ids:
                    reciprocal_rank = 1.0 / rank
                    break

            reciprocal_ranks.append(reciprocal_rank)

            print("\n" + "=" * 70)
            print(f"Category: {case['category']}")
            print(f"Query: {query}")
            print(f"Relevant IDs: {sorted(relevant_ids)}")
            print(f"Retrieved IDs: {ranked_ids}")
            print(f"Relevant hits: {len(hits)}/{len(relevant_ids)}")
            print(f"Reciprocal rank: {reciprocal_rank:.4f}")

        recall = (
            total_retrieved_relevant / total_relevant
            if total_relevant
            else 0.0
        )

        mrr = (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks
            else 0.0
        )

        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Queries evaluated: {len(benchmark)}")
        print(f"Recall@10: {recall:.2%}")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
