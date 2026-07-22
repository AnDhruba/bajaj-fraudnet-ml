from __future__ import annotations

from collections import Counter

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


ENTITY_COLUMNS = {
    "customer_id": "Customer",
    "provider_id": "Provider",
    "intermediary_id": "Intermediary",
    "surveyor_id": "Surveyor",
    "bank_account": "Bank Account",
    "phone": "Phone",
    "address": "Address",
    "device_id": "Device",
    "policy_id": "Policy",
}

NODE_COLORS = {
    "Claim": "#ef4444",
    "Customer": "#2563eb",
    "Provider": "#f59e0b",
    "Intermediary": "#8b5cf6",
    "Surveyor": "#14b8a6",
    "Bank Account": "#22c55e",
    "Phone": "#06b6d4",
    "Address": "#64748b",
    "Device": "#ec4899",
    "Policy": "#84cc16",
}


def _node_id(kind: str, value: str) -> str:
    return f"{kind}::{value}"


def build_claim_graph(claims: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in claims.itertuples(index=False):
        claim_node = _node_id("Claim", str(row.claim_id))
        graph.add_node(
            claim_node,
            label=str(row.claim_id),
            entity_type="Claim",
            confirmed_fraud=int(row.confirmed_fraud),
            outcome_resolved=int(getattr(row, "outcome_resolved", 1)),
            claim_amount=float(row.claim_amount),
            claim_type=str(row.claim_type),
            incident_date=str(pd.Timestamp(row.incident_date).date()),
            route=str(getattr(row, "route", "Not scored")),
            risk_score=float(getattr(row, "risk_score", 0.0)),
        )

        for column, entity_type in ENTITY_COLUMNS.items():
            value = str(getattr(row, column, "NA"))
            if not value or value in {"NA", "None", "nan"}:
                continue
            entity_node = _node_id(entity_type, value)
            if not graph.has_node(entity_node):
                graph.add_node(entity_node, label=value, entity_type=entity_type)
            graph.add_edge(claim_node, entity_node, relation=column)
    return graph


def graph_at_claim_time(claims: pd.DataFrame, claim_id: str) -> nx.Graph:
    match = claims.loc[claims["claim_id"] == claim_id, "incident_date"]
    if match.empty:
        return nx.Graph()
    cutoff = pd.Timestamp(match.iloc[0])
    historical = claims[pd.to_datetime(claims["incident_date"]) <= cutoff].copy()
    if "outcome_date" in historical.columns:
        historical["outcome_resolved"] = (pd.to_datetime(historical["outcome_date"]) <= cutoff).astype(int)
        historical.loc[historical["outcome_resolved"] == 0, "confirmed_fraud"] = 0
    else:
        historical["outcome_resolved"] = 1
    return build_claim_graph(historical)


def extract_neighbourhood(
    graph: nx.Graph,
    claim_id: str,
    radius: int = 2,
    max_nodes: int = 90,
) -> nx.Graph:
    source = _node_id("Claim", claim_id)
    if source not in graph:
        return nx.Graph()
    nodes = list(nx.single_source_shortest_path_length(graph, source, cutoff=radius).keys())
    if len(nodes) > max_nodes:
        remaining = [n for n in nodes if n != source]
        remaining.sort(key=lambda n: graph.degree(n), reverse=True)
        nodes = [source] + remaining[: max_nodes - 1]
    return graph.subgraph(nodes).copy()


def connected_claims(graph: nx.Graph, claim_id: str, radius: int = 2) -> pd.DataFrame:
    source = _node_id("Claim", claim_id)
    if source not in graph:
        return pd.DataFrame()
    distances = nx.single_source_shortest_path_length(graph, source, cutoff=radius)
    records = []
    for node, distance in distances.items():
        attrs = graph.nodes[node]
        if attrs.get("entity_type") == "Claim" and node != source:
            records.append(
                {
                    "claim_id": attrs.get("label"),
                    "distance": distance,
                    "confirmed_fraud": int(attrs.get("confirmed_fraud", 0)),
                    "claim_amount": float(attrs.get("claim_amount", 0)),
                    "route": attrs.get("route", ""),
                    "risk_score": float(attrs.get("risk_score", 0)),
                }
            )
    return pd.DataFrame(records).sort_values(["confirmed_fraud", "risk_score"], ascending=False) if records else pd.DataFrame()


def shared_entity_summary(graph: nx.Graph, claim_id: str) -> dict[str, int]:
    source = _node_id("Claim", claim_id)
    summary: Counter[str] = Counter()
    if source not in graph:
        return dict(summary)
    for entity_node in graph.neighbors(source):
        entity_type = graph.nodes[entity_node].get("entity_type", "Other")
        linked_claims = [
            n for n in graph.neighbors(entity_node)
            if graph.nodes[n].get("entity_type") == "Claim" and n != source
        ]
        if linked_claims:
            summary[entity_type] += len(linked_claims)
    return dict(summary)


def graph_to_plotly(subgraph: nx.Graph, highlighted_claim: str | None = None) -> go.Figure:
    if subgraph.number_of_nodes() == 0:
        return go.Figure()
    pos = nx.spring_layout(subgraph, seed=17, k=0.92, iterations=90)

    edge_x: list[float] = []
    edge_y: list[float] = []
    for source, target in subgraph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    traces: list[go.Scatter] = [
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.7, color="#94a3b8"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )
    ]
    highlighted_node = _node_id("Claim", highlighted_claim) if highlighted_claim else None
    entity_types = sorted({subgraph.nodes[n].get("entity_type", "Other") for n in subgraph.nodes})

    for entity_type in entity_types:
        nodes = [n for n in subgraph.nodes if subgraph.nodes[n].get("entity_type") == entity_type]
        x_values = [pos[n][0] for n in nodes]
        y_values = [pos[n][1] for n in nodes]
        labels = [str(subgraph.nodes[n].get("label", n)) for n in nodes]
        sizes = []
        borders = []
        hover = []
        for node, label in zip(nodes, labels):
            attrs = subgraph.nodes[node]
            degree = subgraph.degree(node)
            sizes.append(31 if node == highlighted_node else min(13 + degree * 1.9, 28))
            borders.append(4 if node == highlighted_node else 1)
            extra = ""
            if entity_type == "Claim":
                extra = (
                    f"<br>Amount: ₹{attrs.get('claim_amount', 0):,.0f}"
                    f"<br>Risk: {attrs.get('risk_score', 0):.1f}/100"
                    f"<br>Route: {attrs.get('route', '')}"
                    f"<br>Outcome known then: {'Yes' if attrs.get('outcome_resolved', 1) else 'No'}"
                    f"<br>Resolved fraud: {'Yes' if attrs.get('confirmed_fraud', 0) else 'No'}"
                )
            hover.append(f"<b>{label}</b><br>Type: {entity_type}<br>Connections: {degree}{extra}")

        traces.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                name=entity_type,
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                hovertext=hover,
                hoverinfo="text",
                marker=dict(
                    size=sizes,
                    color=NODE_COLORS.get(entity_type, "#334155"),
                    line=dict(width=borders, color="#ffffff"),
                    opacity=0.94,
                ),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        margin=dict(l=5, r=5, t=20, b=5),
        height=625,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def highest_risk_entities(claims: pd.DataFrame, column: str, top_n: int = 8) -> pd.DataFrame:
    filtered = claims[~claims[column].astype(str).isin(["NA", "None", "nan", ""])]
    grouped = (
        filtered.groupby(column, dropna=False)
        .agg(
            claims=("claim_id", "count"),
            resolved_fraud=("confirmed_fraud", "sum"),
            exposure=("claim_amount", "sum"),
            average_model_risk=("fraud_probability", "mean"),
            red_claims=("route", lambda s: int((s == "Red").sum())),
        )
        .reset_index()
    )
    grouped["resolved_fraud_rate"] = grouped["resolved_fraud"] / grouped["claims"].clip(lower=1)
    grouped["risk_index"] = (
        grouped["average_model_risk"] * 55
        + grouped["resolved_fraud_rate"] * 35
        + grouped["claims"].clip(upper=25) / 25 * 10
    )
    return grouped.sort_values(["risk_index", "exposure"], ascending=False).head(top_n)


def fraud_communities(claims: pd.DataFrame, min_claims: int = 4) -> pd.DataFrame:
    graph = build_claim_graph(claims)
    records = []
    for component_id, nodes in enumerate(nx.connected_components(graph), start=1):
        claim_nodes = [n for n in nodes if graph.nodes[n].get("entity_type") == "Claim"]
        if len(claim_nodes) < min_claims:
            continue
        claim_ids = [graph.nodes[n].get("label") for n in claim_nodes]
        subset = claims[claims["claim_id"].isin(claim_ids)]
        records.append(
            {
                "community": f"NET-{component_id:03d}",
                "claims": len(subset),
                "resolved_fraud": int(subset["confirmed_fraud"].sum()),
                "exposure": float(subset["claim_amount"].sum()),
                "average_risk": float(subset["risk_score"].mean()),
                "red_claims": int((subset["route"] == "Red").sum()),
                "entity_nodes": len(nodes) - len(claim_nodes),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["resolved_fraud", "exposure"], ascending=False)
