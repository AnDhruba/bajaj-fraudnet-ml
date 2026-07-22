from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import DatasetConfig, generate_claims
from src.graph_engine import (
    build_claim_graph,
    connected_claims,
    extract_neighbourhood,
    fraud_communities,
    graph_at_claim_time,
    graph_to_plotly,
    highest_risk_entities,
    shared_entity_summary,
)
from src.model_engine import FEATURE_LABELS, FraudMLEngine


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "claims.csv"

st.set_page_config(
    page_title="Bajaj FraudNet ML",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 45%, #f8fafc 100%);}
    [data-testid="stSidebar"] {background: #071426;}
    [data-testid="stSidebar"] * {color: #e2e8f0;}
    .hero {
        padding: 1.35rem 1.55rem; border-radius: 18px;
        background: linear-gradient(110deg, #071a33 0%, #12396c 62%, #0f766e 100%);
        color: white; margin-bottom: 1rem; box-shadow: 0 12px 35px rgba(15, 23, 42, .18);
    }
    .hero h1 {font-size: 2.35rem; margin: 0 0 .2rem 0;}
    .hero p {margin: 0; color: #dbeafe; font-size: 1.02rem;}
    .eyebrow {font-size: .74rem; text-transform: uppercase; letter-spacing: .14em; font-weight: 800; color: #67e8f9;}
    .metric-card {
        background: rgba(255,255,255,.96); border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1rem; min-height: 122px; box-shadow: 0 5px 18px rgba(15,23,42,.06);
    }
    .metric-label {font-size: .78rem; color: #64748b; font-weight: 800; text-transform: uppercase; letter-spacing:.045em;}
    .metric-value {font-size: 1.62rem; color: #0f172a; font-weight: 850; margin-top:.28rem;}
    .metric-note {font-size: .77rem; color: #64748b; margin-top:.22rem;}
    .risk-high {background:#fff1f2;border-left:6px solid #e11d48;padding:1rem;border-radius:10px;}
    .risk-medium {background:#fffbeb;border-left:6px solid #f59e0b;padding:1rem;border-radius:10px;}
    .risk-low {background:#ecfdf5;border-left:6px solid #10b981;padding:1rem;border-radius:10px;}
    .reason-card {background:white;border:1px solid #e2e8f0;border-radius:12px;padding:.82rem;margin-bottom:.55rem;}
    .protect-card {background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:.82rem;margin-bottom:.55rem;}
    .warn-card {background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:.82rem;margin-bottom:.55rem;}
    .small-muted {font-size:.82rem;color:#64748b;}
    div[data-testid="stMetric"] {background:rgba(255,255,255,.94);border:1px solid #e2e8f0;padding:12px;border-radius:12px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_claims() -> pd.DataFrame:
    expected_columns = {
        "outcome_date",
        "sum_insured",
        "damage_consistency",
        "document_verification_score",
        "fraud_pattern",
    }
    if DATA_PATH.exists():
        frame = pd.read_csv(DATA_PATH, parse_dates=["incident_date", "outcome_date"])
        if expected_columns.issubset(frame.columns):
            return frame

    frame = generate_claims(DatasetConfig())
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    return frame


@st.cache_resource(show_spinner="Training time-safe fraud models…")
def build_system(data_signature: tuple[int, str, str]):
    del data_signature
    raw_claims = load_claims()
    engine = FraudMLEngine(raw_claims)
    scored = engine.claims.copy()
    graph = build_claim_graph(scored)
    return engine, scored, graph


def currency(value: float) -> str:
    value = float(value)
    if value >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"
    if value >= 100_000:
        return f"₹{value / 100_000:.1f} L"
    return f"₹{value:,.0f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def risk_gauge(score: float, red_threshold: float, amber_threshold: float) -> go.Figure:
    red = red_threshold * 100
    amber = amber_threshold * 100
    color = "#e11d48" if score >= red else "#f59e0b" if score >= amber else "#10b981"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.34},
                "steps": [
                    {"range": [0, amber], "color": "#dcfce7"},
                    {"range": [amber, red], "color": "#fef3c7"},
                    {"range": [red, 100], "color": "#ffe4e6"},
                ],
                "threshold": {"line": {"color": "#0f172a", "width": 3}, "value": score},
            },
            title={"text": "Calibrated Fraud Probability"},
        )
    )
    fig.update_layout(height=305, margin=dict(l=25, r=25, t=65, b=15), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def claim_selector_options(frame: pd.DataFrame) -> dict[str, str]:
    def first(pattern: str, newest: bool = True) -> str:
        subset = frame[frame["fraud_pattern"] == pattern].sort_values("incident_date", ascending=not newest)
        return str(subset.iloc[0]["claim_id"])

    ordinary = frame[(frame["fraud_pattern"] == "NONE") & (frame["route"] == "Green")].sort_values("incident_date", ascending=False)
    return {
        "Coordinated motor ring — hidden shared infrastructure": first("MOTOR_COLLUSION_RING"),
        "Hospital–TPA collusion ring": first("HEALTH_COLLUSION_RING"),
        "Subtle fraud — normal amount, rotating identities": first("SUBTLE_BEHAVIOURAL_RING"),
        "Unusual but genuine — protected from false positive": first("UNUSUAL_GENUINE"),
        "Ordinary low-risk claim": str(ordinary.iloc[0]["claim_id"]),
    }


def display_reasons(reasons: Iterable[dict], protective: bool = False) -> None:
    items = list(reasons)
    if not items:
        st.caption("No dominant reason isolated; the decision reflects several smaller interactions.")
        return
    css = "protect-card" if protective else "reason-card"
    sign = "Protective" if protective else "Risk contribution"
    for item in items:
        contribution = abs(float(item.get("contribution", 0))) * 100
        st.markdown(
            f"<div class='{css}'><b>{item['label']}</b><br>"
            f"<span class='small-muted'>{sign}: approximately {contribution:.1f} probability points in this local explanation.</span></div>",
            unsafe_allow_html=True,
        )


def outcome_badge(route: str, reason: str) -> None:
    css = {"Red": "risk-high", "Amber": "risk-medium", "Green": "risk-low"}[route]
    action = {
        "Red": "Pause straight-through settlement and assign SIU review",
        "Amber": "Request targeted evidence or limited human verification",
        "Green": "Fast-track with standard controls",
    }[route]
    st.markdown(
        f"<div class='{css}'><b>{route} lane — {action}</b><br><span class='small-muted'>{reason}</span></div>",
        unsafe_allow_html=True,
    )


def claim_summary_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Claim ID", row["claim_id"]],
            ["Claim type", row["claim_type"]],
            ["Incident date", pd.Timestamp(row["incident_date"]).date().isoformat()],
            ["Claim amount", currency(row["claim_amount"])],
            ["Sum insured", currency(row["sum_insured"])],
            ["Provider", row["provider_id"]],
            ["Surveyor / TPA", row["surveyor_id"] if row["surveyor_id"] != "NA" else row["intermediary_id"]],
            ["City", row["city"]],
            ["Model split", row["split"]],
        ],
        columns=["Field", "Value"],
    )


raw_claims = load_claims()
signature = (
    len(raw_claims),
    str(pd.to_datetime(raw_claims["incident_date"]).min()),
    str(pd.to_datetime(raw_claims["incident_date"]).max()),
)
engine, claims, graph = build_system(signature)
scenario_options = claim_selector_options(claims)

st.sidebar.markdown("## 🕸️ Bajaj FraudNet")
st.sidebar.caption("Real-time, ML-powered ecosystem fraud intelligence")
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Command Centre",
        "FNOL Risk Scanner",
        "False-Positive Lab",
        "Network Investigator",
        "Model Performance",
        "Enterprise Blueprint",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Demonstrator dataset**")
st.sidebar.write(f"{len(claims):,} synthetic claims")
st.sidebar.write(f"{graph.number_of_nodes():,} graph entities")
st.sidebar.write(f"{graph.number_of_edges():,} relationships")
st.sidebar.write(f"Train / validation / test: 65% / 17% / 18%")
st.sidebar.caption("Synthetic records only. Model metrics demonstrate methodology, not expected production performance.")

st.markdown(
    """
<div class="hero">
    <div class="eyebrow">Bajaj General Insurance • ATOM Season 9</div>
    <h1>FraudNet ML</h1>
    <p>Detect coordinated fraud at FNOL while protecting genuine customers whose claims are unusual but legitimate.</p>
</div>
""",
    unsafe_allow_html=True,
)


if page == "Executive Command Centre":
    test = claims[claims["split"] == "test"]
    ops = engine.operational_metrics
    total_exposure = float(claims["claim_amount"].sum())
    red_exposure = float(claims.loc[claims["route"] == "Red", "claim_amount"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Claims analysed", f"{len(claims):,}", "Time-ordered historical and live simulation")
    with c2:
        metric_card("Fraud recall", pct(engine.model_metrics["recall"]), "Red-lane detection on unseen test claims")
    with c3:
        metric_card("Genuine fast-track", pct(ops["genuine_fast_track_rate"]), "Test-set genuine claims routed Green")
    with c4:
        metric_card("Unusual genuine protected", pct(ops["unusual_genuine_fast_track_rate"]), "Explicit false-positive stress cases")
    with c5:
        metric_card("Exposure analysed", currency(total_exposure), f"Red-lane exposure: {currency(red_exposure)}")

    st.subheader("Portfolio routing and model value")
    left, right = st.columns([1.1, 1])
    with left:
        route_counts = claims["route"].value_counts().reindex(["Green", "Amber", "Red"], fill_value=0).reset_index()
        route_counts.columns = ["Route", "Claims"]
        fig = px.bar(
            route_counts,
            x="Route",
            y="Claims",
            text="Claims",
            color="Route",
            color_discrete_map={"Green": "#10b981", "Amber": "#f59e0b", "Red": "#e11d48"},
            title="Green / Amber / Red operating lanes",
        )
        fig.update_layout(showlegend=False, height=365, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        comparison = engine.comparison_table().copy()
        long = comparison.melt(
            id_vars="Approach",
            value_vars=["Recall", "Fraud value captured"],
            var_name="Metric",
            value_name="Value",
        )
        fig = px.bar(long, x="Metric", y="Value", color="Approach", barmode="group", text_auto=".0%", title="Rules versus learning system")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        fig.update_layout(height=365, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.subheader("Highest-risk providers")
        provider = highest_risk_entities(claims, "provider_id", 8).copy()
        provider["exposure"] = provider["exposure"].map(currency)
        provider["resolved_fraud_rate"] = provider["resolved_fraud_rate"].map(pct)
        provider["average_model_risk"] = provider["average_model_risk"].map(pct)
        st.dataframe(
            provider[["provider_id", "claims", "resolved_fraud", "resolved_fraud_rate", "average_model_risk", "exposure"]],
            hide_index=True,
            use_container_width=True,
        )
    with b:
        st.subheader("Unseen test-set operating outcomes")
        route_truth = pd.crosstab(test["route"], test["confirmed_fraud"]).rename(columns={0: "Genuine", 1: "Fraud"})
        route_truth = route_truth.reindex(["Green", "Amber", "Red"], fill_value=0)
        st.dataframe(route_truth, use_container_width=True)
        st.info(
            "Anomaly detection may move a novel claim to Amber for evidence collection, but it cannot independently create a Red fraud verdict."
        )


elif page == "FNOL Risk Scanner":
    st.subheader("Score a claim at First Notice of Loss")
    st.caption("The prediction uses only features and resolved outcomes that would have been available on the claim's incident date.")

    scenario = st.selectbox("Demonstration scenario", list(scenario_options))
    selected_claim = scenario_options[scenario]
    result = engine.score_claim(selected_claim)
    row = claims.loc[claims["claim_id"] == selected_claim].iloc[0]

    left, middle, right = st.columns([1.2, 1, 1])
    with left:
        st.plotly_chart(risk_gauge(result.risk_score, engine.red_threshold, engine.amber_threshold), use_container_width=True)
    with middle:
        st.metric("Graph risk", f"{result.graph_risk_score:.0f}/100")
        st.metric("Anomaly percentile", pct(result.anomaly_percentile))
        st.metric("Rule score", f"{result.rule_score:.0f}/100")
    with right:
        st.metric("Legitimacy evidence", f"{result.legitimacy_score:.0f}/100")
        st.metric("Claim amount", currency(row["claim_amount"]))
        st.metric("Claim-to-cover ratio", pct(row["claim_to_sum_insured"]))

    outcome_badge(result.route, result.route_reason)

    detail_left, detail_right = st.columns([0.9, 1.1])
    with detail_left:
        st.markdown("#### Claim context")
        st.dataframe(claim_summary_table(row), hide_index=True, use_container_width=True)
        with st.expander("Demonstrator ground truth", expanded=False):
            st.write("**Synthetic label:**", "Confirmed fraud" if row["confirmed_fraud"] else "Genuine")
            st.write("**Scenario type:**", row["fraud_pattern"])
            st.write("**Case note:**", row["case_note"])
            st.caption("Ground truth is shown only for judging the demonstrator; it is not available to the live model at FNOL.")

    with detail_right:
        risk_col, protect_col = st.columns(2)
        with risk_col:
            st.markdown("#### Why risk increased")
            display_reasons(result.positive_reasons, protective=False)
        with protect_col:
            st.markdown("#### Why false-positive risk decreased")
            display_reasons(result.protective_reasons, protective=True)

    st.markdown("#### Time-safe ecosystem view")
    historical_graph = graph_at_claim_time(claims, selected_claim)
    neighbourhood = extract_neighbourhood(historical_graph, selected_claim, radius=2, max_nodes=85)
    st.plotly_chart(graph_to_plotly(neighbourhood, selected_claim), use_container_width=True)
    if result.connected_known_fraud_claims:
        st.warning(f"Resolved fraud links known at FNOL: {', '.join(result.connected_known_fraud_claims[:8])}")
    else:
        st.success("No directly connected resolved-fraud claim was known at FNOL; the model relied on learned behavioural and contextual evidence.")


elif page == "False-Positive Lab":
    st.subheader("Can the model distinguish unusual genuine claims from subtle fraud?")
    st.caption("This page stress-tests the exact weakness of fixed thresholds: unusual does not automatically mean fraudulent.")

    unusual_id = scenario_options["Unusual but genuine — protected from false positive"]
    subtle_id = scenario_options["Subtle fraud — normal amount, rotating identities"]
    unusual = claims.loc[claims["claim_id"] == unusual_id].iloc[0]
    subtle = claims.loc[claims["claim_id"] == subtle_id].iloc[0]
    unusual_result = engine.score_claim(unusual_id)
    subtle_result = engine.score_claim(subtle_id)

    left, right = st.columns(2)
    with left:
        st.markdown("### Unusual but genuine")
        outcome_badge(unusual_result.route, unusual_result.route_reason)
        x1, x2, x3 = st.columns(3)
        x1.metric("ML risk", pct(unusual_result.fraud_probability))
        x2.metric("Anomaly", pct(unusual_result.anomaly_percentile))
        x3.metric("Legitimacy", f"{unusual_result.legitimacy_score:.0f}/100")
        st.write(unusual["case_note"])
        st.markdown("**Protective context used by the model**")
        display_reasons(unusual_result.protective_reasons, protective=True)

    with right:
        st.markdown("### Subtle fraud designed to evade rules")
        outcome_badge(subtle_result.route, subtle_result.route_reason)
        x1, x2, x3 = st.columns(3)
        x1.metric("ML risk", pct(subtle_result.fraud_probability))
        x2.metric("Rule score", f"{subtle_result.rule_score:.0f}/100")
        x3.metric("Graph risk", f"{subtle_result.graph_risk_score:.0f}/100")
        st.write(subtle["case_note"])
        st.markdown("**Learned behavioural and network evidence**")
        display_reasons(subtle_result.positive_reasons, protective=False)

    compare = pd.DataFrame(
        [
            {
                "Scenario": "Unusual genuine",
                "Fraud probability": unusual_result.fraud_probability,
                "Rule score": unusual_result.rule_score / 100,
                "Anomaly percentile": unusual_result.anomaly_percentile,
                "Legitimacy score": unusual_result.legitimacy_score / 100,
            },
            {
                "Scenario": "Subtle fraud",
                "Fraud probability": subtle_result.fraud_probability,
                "Rule score": subtle_result.rule_score / 100,
                "Anomaly percentile": subtle_result.anomaly_percentile,
                "Legitimacy score": subtle_result.legitimacy_score / 100,
            },
        ]
    )
    chart = compare.melt(id_vars="Scenario", var_name="Signal", value_name="Value")
    fig = px.bar(chart, x="Signal", y="Value", color="Scenario", barmode="group", text_auto=".0%", title="The system evaluates context, not one unusual threshold")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "Design principle: high anomaly alone can trigger targeted verification, but only learned fraud probability or decisive evidence conflict can create a Red route."
    )


elif page == "Network Investigator":
    st.subheader("Interactive ecosystem investigation")
    high_first = claims.sort_values(["risk_score", "incident_date"], ascending=False)["claim_id"].tolist()
    c1, c2, c3 = st.columns([1.25, 1, 1])
    with c1:
        selected_claim = st.selectbox("Start from claim", high_first)
    with c2:
        radius = st.slider("Relationship depth", 1, 3, 2)
    with c3:
        max_nodes = st.slider("Maximum nodes", 25, 120, 80, step=5)

    result = engine.score_claim(selected_claim)
    historical_graph = graph_at_claim_time(claims, selected_claim)
    network = extract_neighbourhood(historical_graph, selected_claim, radius=radius, max_nodes=max_nodes)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Network entities", network.number_of_nodes())
    m2.metric("Relationships", network.number_of_edges())
    m3.metric("Known-fraud links", len(result.connected_known_fraud_claims))
    m4.metric("Calibrated risk", f"{result.risk_score:.0f}/100")

    st.plotly_chart(graph_to_plotly(network, selected_claim), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Shared-entity evidence")
        shared = shared_entity_summary(historical_graph, selected_claim)
        if shared:
            shared_df = pd.DataFrame([{"Entity type": k, "Other linked claims": v} for k, v in shared.items()])
            st.dataframe(shared_df.sort_values("Other linked claims", ascending=False), hide_index=True, use_container_width=True)
        else:
            st.success("No entity is shared with another claim at this point in time.")
    with right:
        st.markdown("#### Connected claims")
        linked = connected_claims(historical_graph, selected_claim, radius=2)
        if linked.empty:
            st.success("No connected claims found within two graph hops.")
        else:
            linked["claim_amount"] = linked["claim_amount"].map(currency)
            st.dataframe(linked.head(20), hide_index=True, use_container_width=True)

    st.markdown("#### Investigator workflow")
    a1, a2, a3, a4 = st.columns(4)
    a1.button("Place settlement hold", use_container_width=True)
    a2.button("Request targeted evidence", use_container_width=True)
    a3.button("Expand provider review", use_container_width=True)
    a4.button("Escalate network case", use_container_width=True)


elif page == "Model Performance":
    st.subheader("Does the learning system outperform fixed rules?")
    st.caption("All results below use the chronologically later, unseen test period. Synthetic metrics validate the prototype methodology only.")

    m = engine.model_metrics
    o = engine.operational_metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", pct(m["precision"]))
    c2.metric("Recall", pct(m["recall"]))
    c3.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    c4.metric("False-positive rate", pct(m["false_positive_rate"]))
    c5.metric("Fraud value captured", pct(m["fraud_value_capture"]))

    comparison = engine.comparison_table().copy()
    for col in ["Precision", "Recall", "False-positive rate", "Fraud value captured"]:
        comparison[col] = comparison[col].map(pct)
    comparison["PR-AUC"] = comparison["PR-AUC"].map(lambda x: f"{x:.3f}")
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        cm = pd.DataFrame(
            engine.confusion,
            index=["Actual genuine", "Actual fraud"],
            columns=["Predicted genuine", "Predicted fraud"],
        )
        fig = px.imshow(cm, text_auto=True, aspect="auto", title="Red-lane confusion matrix")
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        curve = engine.pr_curve.dropna(subset=["threshold"])
        fig = px.line(curve, x="recall", y="precision", title="Precision–recall trade-off")
        fig.add_scatter(
            x=[m["recall"]],
            y=[m["precision"]],
            mode="markers+text",
            text=["Chosen Red route"],
            textposition="top center",
            marker={"size": 12},
            showlegend=False,
        )
        fig.update_xaxes(range=[0, 1.02], tickformat=".0%")
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        calibration = engine.calibration_data.copy()
        fig = px.scatter(calibration, x="predicted", y="observed", title="Probability calibration")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash"})
        fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Predicted fraud probability")
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Observed fraud rate")
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        importance = engine.feature_importance.head(12).sort_values("importance")
        fig = px.bar(importance, x="importance", y="label", orientation="h", title="Model feature importance on unseen data")
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10), yaxis_title="", xaxis_title="Permutation importance")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Time-based split prevents future leakage")
    splits = engine.split_summary().copy()
    splits["start_date"] = pd.to_datetime(splits["start_date"]).dt.date.astype(str)
    splits["end_date"] = pd.to_datetime(splits["end_date"]).dt.date.astype(str)
    st.dataframe(splits, hide_index=True, use_container_width=True)

    st.markdown("#### Customer-protection operating metrics")
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Review recall", pct(o["review_recall"]))
    x2.metric("Review precision", pct(o["review_precision"]))
    x3.metric("Genuine fast-track", pct(o["genuine_fast_track_rate"]))
    x4.metric("Unusual genuine fast-track", pct(o["unusual_genuine_fast_track_rate"]))

    st.info(
        "Accuracy is intentionally not the headline metric. Fraud is imbalanced, so the prototype tracks precision, recall, PR-AUC, false-positive rate, rupee-value capture and genuine-customer fast-tracking."
    )


else:
    st.subheader("Prototype architecture and production roadmap")

    st.markdown(
        """
### What this prototype genuinely implements

1. **Time-safe feature engineering:** every claim uses only prior FNOL records and fraud outcomes resolved before that claim date.  
2. **Supervised behavioural ML:** a gradient-boosted classifier learns non-linear combinations associated with confirmed fraud.  
3. **Probability calibration:** a separate validation period converts model output into a more interpretable fraud probability.  
4. **Graph intelligence:** shared accounts, devices, vendors, surveyors, TPAs and known-fraud neighbourhoods become model features.  
5. **Unsupervised anomaly detection:** Isolation Forest searches for novel behaviour, but anomaly alone cannot label a customer fraudulent.  
6. **Deterministic controls:** decisive evidence conflicts can override normal routing.  
7. **Legitimacy evidence:** verified location, coherent damage, strong documents, catastrophe context and customer tenure help prevent false positives.  
8. **Chronological evaluation:** train, validation and test periods are separated to avoid evaluating on seen claims.
"""
    )

    architecture = pd.DataFrame(
        [
            ["Experience", "Streamlit command centre and investigator interface", "Enterprise web/mobile interface"],
            ["Data ingestion", "Synthetic historical claims and FNOL simulation", "Claims, policy, payment, vendor, document and device APIs"],
            ["Entity resolution", "Exact synthetic identifiers", "Probabilistic matching for aliases, spelling variation and shared identities"],
            ["Graph", "NetworkX time-safe claim graph", "Neo4j / managed graph database with streaming updates"],
            ["Supervised ML", "HistGradientBoosting + calibration", "Champion/challenger gradient boosting and cost-sensitive learning"],
            ["Novelty detection", "Isolation Forest with protected routing", "Segment-specific anomaly models and drift detection"],
            ["Graph ML", "Historical graph features", "Node2Vec / GraphSAGE / link prediction after sufficient real labels"],
            ["Governance", "Local explanations and visible thresholds", "RBAC, audit trail, model registry, appeals and model-risk committee"],
        ],
        columns=["Layer", "Current demonstrator", "Production target"],
    )
    st.dataframe(architecture, hide_index=True, use_container_width=True)

    st.markdown("### Improvements to be implemented as next steps")
    roadmap = pd.DataFrame(
        [
            ["1. Bajaj data pilot", "Train on de-identified historical claims and SIU outcomes", "Replace proxy performance with real baseline and segment-specific calibration"],
            ["2. Entity resolution", "Match aliases, phones, bank accounts, addresses and devices probabilistically", "Reveal networks fragmented by spelling changes or rotated identities"],
            ["3. False-positive optimisation", "Add verified legitimacy signals and tune thresholds to SIU capacity", "Fast-track more genuine claims without sacrificing fraud value capture"],
            ["4. Human feedback loop", "Feed investigator decisions, appeals and recoveries back into training", "Continuously learn from confirmed outcomes and corrected alerts"],
            ["5. Drift monitoring", "Track data, prediction and performance drift by product and region", "Detect fraud adaptation and trigger champion/challenger retraining"],
            ["6. Graph embeddings / GNN", "Learn indirect network similarity once sufficient labelled graph history exists", "Identify collusion even when direct identifiers rotate"],
            ["7. Production governance", "Bias tests, audit trails, reason codes and mandatory human review for adverse decisions", "Reliable, explainable and compliant claim handling"],
        ],
        columns=["Next step", "Implementation", "Business value"],
    )
    st.dataframe(roadmap, hide_index=True, use_container_width=True)

    st.markdown("### Enterprise rollout")
    rollout = pd.DataFrame(
        [
            ["0–3 months", "Historical motor-claims pilot", "Rules baseline, entity resolution, time-safe ML benchmark"],
            ["3–6 months", "Shadow-mode FNOL scoring", "No automated adverse decisions; compare with SIU outcomes"],
            ["6–9 months", "Green-lane automation", "Fast-track low-risk claims while Amber/Red remain human-reviewed"],
            ["9–15 months", "Health, garage, TPA and payment ecosystem expansion", "Cross-product network intelligence"],
            ["15–24 months", "Graph embeddings, continuous learning and case prioritisation", "Earlier fraud-ring discovery at enterprise scale"],
        ],
        columns=["Period", "Release", "Control and outcome"],
    )
    st.dataframe(rollout, hide_index=True, use_container_width=True)

    st.warning(
        "Production principle: the model prioritises and explains risk; it should not automatically reject a claim solely because it is anomalous. High-impact adverse decisions remain subject to evidence and human review."
    )

st.markdown("---")
st.caption("Bajaj FraudNet ML • ATOM Season 9 demonstrator • Synthetic data only • Not a production claim-adjudication system")
