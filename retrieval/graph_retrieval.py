"""Lightweight co-occurrence entity graph baseline.

This intentionally uses a transparent heuristic rather than claiming to perform
LLM-grade knowledge-graph extraction. It links capitalized terms that co-occur
inside a document chunk and retrieves chunks connected to query terms.
"""
import re
from collections import defaultdict

ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9&-]*(?:\s+[A-Z][a-zA-Z0-9&-]*){0,3}\b")

class CooccurrenceGraph:
    def __init__(self, documents):
        self.documents = documents
        self.entity_to_docs = defaultdict(set)
        self.edges = defaultdict(set)
        for idx, doc in enumerate(documents):
            entities = {e.lower() for e in ENTITY_RE.findall(doc.text) if len(e) > 1}
            for entity in entities:
                self.entity_to_docs[entity].add(idx)
            entities = list(entities)
            for i, a in enumerate(entities):
                for b in entities[i+1:]:
                    self.edges[a].add(b)
                    self.edges[b].add(a)

    def search(self, query, k=5):
        terms = {e.lower() for e in ENTITY_RE.findall(query)}
        terms.update(t.lower() for t in re.findall(r"\b[A-Z][a-zA-Z0-9&-]*\b", query))
        expanded = set(terms)
        for term in list(terms):
            expanded.update(self.edges.get(term, set()))
        scores = defaultdict(float)
        for entity in expanded:
            for idx in self.entity_to_docs.get(entity, set()):
                scores[idx] += 2.0 if entity in terms else 1.0
        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        return [{"document": self.documents[i], "score": scores[i], "method": "cooccurrence_graph"} for i in ranked]
