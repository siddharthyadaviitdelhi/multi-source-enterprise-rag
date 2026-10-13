from ingestion.common import Document
from retrieval.sentence_window import make_sentence_windows, restore_window
from retrieval.graph_retrieval import CooccurrenceGraph

def test_sentence_window_restores_context():
    docs = [Document("Alpha works with Beta. Beta builds systems. Gamma is separate.", "x", "text", {})]
    windows = make_sentence_windows(docs, window=1)
    assert len(windows) == 3
    assert "Beta builds systems." in windows[0].metadata["window_text"]
    restored = restore_window({"document": windows[0], "score": 0.5})
    assert "Alpha works with Beta." in restored["document"].text

def test_graph_retrieval_returns_cooccurring_entities():
    docs = [
        Document("Alice manages the Orion Platform. Orion Platform supports reports.", "a", "text", {}),
        Document("Finance tracks quarterly budgets.", "b", "text", {})
    ]
    graph = CooccurrenceGraph(docs)
    results = graph.search("Orion Platform", k=2)
    assert results
    assert results[0]["document"].source == "a"
