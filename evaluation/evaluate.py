"""Small retrieval smoke evaluation. Reports evidence retrieval, not model accuracy."""
import json
from pathlib import Path
from ingestion.common import Document, chunk_document
from rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]

def load_sample_docs():
    docs = []
    for filename in ["sample_policy.csv", "sample_handbook.csv"]:
        from ingestion.csv_loader import load_csv
        docs.extend(load_csv(ROOT / "data" / "sample" / filename))
    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d, 900, 150))
    return chunks

def main():
    pipeline = RAGPipeline(load_sample_docs())
    tests = json.loads((ROOT / "evaluation" / "test_questions.json").read_text())
    passed = 0
    for case in tests:
        results = pipeline.retrieve(case["question"], mode="hybrid", top_k=5, use_reranker=False)
        text = "\n".join(r["document"].text for r in results).lower()
        source_names = " ".join(r["document"].source for r in results)
        ok = case["expected_keyword"].lower() in text and case["expected_source_contains"] in source_names
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} | {case['question']}")
        if not ok:
            print("  Retrieved sources:", source_names)
    print(f"\nRetrieval smoke-test result: {passed}/{len(tests)}")
    print("This is a tiny hand-written smoke test, not a statistically meaningful benchmark.")

if __name__ == "__main__":
    main()
