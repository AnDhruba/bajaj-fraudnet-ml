from __future__ import annotations

from src.data_generator import generate_claims
from src.graph_engine import build_claim_graph
from src.model_engine import FraudMLEngine

claims = generate_claims()
engine = FraudMLEngine(claims)
graph = build_claim_graph(engine.claims)

assert len(engine.claims) > 1_000
assert graph.number_of_nodes() > len(engine.claims)
assert engine.model_metrics["recall"] >= 0.70
assert engine.operational_metrics["unusual_genuine_fast_track_rate"] >= 0.70
assert engine.claims[engine.claims["fraud_pattern"] == "SUBTLE_BEHAVIOURAL_RING"]["route"].isin(["Amber", "Red"]).mean() >= 0.70

print("Smoke test passed.")
print(f"Claims: {len(engine.claims):,}")
print(f"Graph: {graph.number_of_nodes():,} nodes / {graph.number_of_edges():,} edges")
print(f"Test recall: {engine.model_metrics['recall']:.1%}")
print(f"Test false-positive rate: {engine.model_metrics['false_positive_rate']:.1%}")
print(f"Unusual genuine fast-track: {engine.operational_metrics['unusual_genuine_fast_track_rate']:.1%}")
