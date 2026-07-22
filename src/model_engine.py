from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler


FEATURE_LABELS = {
    "claim_amount_log": "Claim amount",
    "claim_to_sum_insured": "Claim-to-cover ratio",
    "amount_peer_z": "Amount versus historical peers",
    "incident_severity": "Incident severity",
    "night_incident": "Incident timing",
    "catastrophe_event": "Known catastrophe context",
    "customer_tenure_log": "Customer relationship tenure",
    "policy_tenure_log": "Policy tenure",
    "verified_location": "Verified incident location",
    "damage_consistency": "Damage-to-narrative consistency",
    "document_verification_score": "Document verification confidence",
    "document_mismatch": "Document mismatch",
    "duplicate_media": "Duplicate media evidence",
    "provider_prior_claims_log": "Provider history volume",
    "provider_prior_fraud_rate": "Provider prior fraud rate",
    "surveyor_prior_claims_log": "Surveyor history volume",
    "surveyor_prior_fraud_rate": "Surveyor prior fraud rate",
    "intermediary_prior_claims_log": "TPA/intermediary history volume",
    "intermediary_prior_fraud_rate": "TPA/intermediary prior fraud rate",
    "bank_prior_claims_log": "Bank-account reuse",
    "device_prior_claims_log": "Device reuse",
    "phone_prior_claims_log": "Phone reuse",
    "address_prior_claims_log": "Address reuse",
    "customer_prior_claims_log": "Customer claim history",
    "customer_prior_fraud_rate": "Customer prior fraud rate",
    "pair_prior_claims_log": "Provider-surveyor/TPA relationship frequency",
    "pair_prior_fraud_rate": "Relationship prior fraud rate",
    "linked_prior_claims_log": "Network-linked historical claims",
    "connected_known_fraud_log": "Connections to resolved fraud",
    "network_prior_fraud_rate": "Neighbourhood prior fraud rate",
    "shared_entity_types": "Number of reused entity types",
    "relationship_reuse_score": "Relationship reuse intensity",
    "claim_type_motor": "Motor-claim indicator",
    "vehicle_segment_code": "Vehicle segment",
}

MODEL_FEATURES = list(FEATURE_LABELS)
ANOMALY_FEATURES = [
    "claim_amount_log",
    "claim_to_sum_insured",
    "amount_peer_z",
    "incident_severity",
    "night_incident",
    "provider_prior_claims_log",
    "surveyor_prior_claims_log",
    "intermediary_prior_claims_log",
    "bank_prior_claims_log",
    "device_prior_claims_log",
    "phone_prior_claims_log",
    "address_prior_claims_log",
    "linked_prior_claims_log",
    "shared_entity_types",
    "relationship_reuse_score",
    "damage_consistency",
    "document_verification_score",
]

VEHICLE_SEGMENT_CODE = {"NA": 0, "Economy": 1, "Mid": 2, "Premium": 3, "Luxury": 4}


@dataclass
class RiskResult:
    claim_id: str
    fraud_probability: float
    risk_score: float
    risk_band: str
    route: str
    route_reason: str
    anomaly_percentile: float
    graph_risk_score: float
    rule_score: float
    legitimacy_score: float
    positive_reasons: list[dict[str, Any]]
    protective_reasons: list[dict[str, Any]]
    connected_known_fraud_claims: list[str]
    hard_override: bool


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(math.sqrt(self.m2 / (self.n - 1)), 1.0)


def _entity_values(row: pd.Series) -> dict[str, str]:
    values = {
        "provider": str(row["provider_id"]),
        "bank": str(row["bank_account"]),
        "phone": str(row["phone"]),
        "address": str(row["address"]),
        "device": str(row["device_id"]),
        "customer": str(row["customer_id"]),
    }
    if str(row["surveyor_id"]) not in {"NA", "None", "nan", ""}:
        values["surveyor"] = str(row["surveyor_id"])
    if str(row["intermediary_id"]) not in {"NA", "None", "nan", ""}:
        values["intermediary"] = str(row["intermediary_id"])
    return values


def _pair_key(row: pd.Series) -> str:
    secondary = str(row["surveyor_id"])
    if secondary in {"NA", "None", "nan", ""}:
        secondary = str(row["intermediary_id"])
    return f"{row['provider_id']}|{secondary}"


def _smoothed_rate(fraud: int, total: int, prior: float = 0.035, strength: float = 18.0) -> float:
    return float((fraud + prior * strength) / (total + strength))


def build_time_safe_features(claims: pd.DataFrame) -> pd.DataFrame:
    """Build features using only information available at each claim's incident time.

    Claim counts become available immediately after FNOL. Fraud outcomes only become
    available after the synthetic outcome_date, preventing future-label leakage.
    """
    df = claims.copy().sort_values(["incident_date", "claim_id"]).reset_index(drop=True)
    for col in ["incident_date", "outcome_date"]:
        df[col] = pd.to_datetime(df[col])

    entity_claim_counts: dict[tuple[str, str], int] = defaultdict(int)
    entity_resolved_counts: dict[tuple[str, str], int] = defaultdict(int)
    entity_resolved_fraud: dict[tuple[str, str], int] = defaultdict(int)
    entity_claim_ids: dict[tuple[str, str], list[str]] = defaultdict(list)

    pair_claim_counts: dict[str, int] = defaultdict(int)
    pair_resolved_counts: dict[str, int] = defaultdict(int)
    pair_resolved_fraud: dict[str, int] = defaultdict(int)

    peer_stats: dict[str, RunningStats] = defaultdict(RunningStats)
    claim_label_by_id: dict[str, int] = {}
    known_fraud_claims: set[str] = set()
    outcome_heap: list[tuple[pd.Timestamp, str, int, dict[str, str], str]] = []

    feature_rows: list[dict[str, Any]] = []

    def release_outcomes(current_time: pd.Timestamp) -> None:
        while outcome_heap and outcome_heap[0][0] <= current_time:
            _, resolved_claim_id, label, entities, pair = heappop(outcome_heap)
            for kind, value in entities.items():
                key = (kind, value)
                entity_resolved_counts[key] += 1
                entity_resolved_fraud[key] += int(label)
            pair_resolved_counts[pair] += 1
            pair_resolved_fraud[pair] += int(label)
            claim_label_by_id[resolved_claim_id] = int(label)
            if label:
                known_fraud_claims.add(resolved_claim_id)

    for _, row in df.iterrows():
        release_outcomes(pd.Timestamp(row["incident_date"]))
        entities = _entity_values(row)
        pair = _pair_key(row)

        linked_claim_ids: set[str] = set()
        linked_entity_types = 0
        relationship_reuse = 0.0
        for kind, value in entities.items():
            key = (kind, value)
            previous = entity_claim_ids[key]
            if previous:
                linked_entity_types += 1
                linked_claim_ids.update(previous)
                relationship_reuse += math.log1p(len(previous))

        connected_known = sorted(linked_claim_ids.intersection(known_fraud_claims))
        linked_resolved = [cid for cid in linked_claim_ids if cid in claim_label_by_id]
        linked_fraud = sum(claim_label_by_id[cid] for cid in linked_resolved)

        claim_type = str(row["claim_type"])
        amount = float(row["claim_amount"])
        stats = peer_stats[claim_type]
        amount_peer_z = 0.0 if stats.n < 20 else (amount - stats.mean) / stats.std
        amount_peer_z = float(np.clip(amount_peer_z, -4.0, 8.0))

        def count(kind: str) -> int:
            value = entities.get(kind)
            return entity_claim_counts[(kind, value)] if value is not None else 0

        def rate(kind: str) -> float:
            value = entities.get(kind)
            if value is None:
                return 0.035
            key = (kind, value)
            return _smoothed_rate(entity_resolved_fraud[key], entity_resolved_counts[key])

        feature = {
            "claim_id": str(row["claim_id"]),
            "claim_amount_log": math.log1p(amount),
            "claim_to_sum_insured": float(amount / max(float(row["sum_insured"]), 1.0)),
            "amount_peer_z": amount_peer_z,
            "incident_severity": float(row["incident_severity"]),
            "night_incident": float(0 <= int(row["incident_hour"]) <= 4),
            "catastrophe_event": float(row["catastrophe_event"]),
            "customer_tenure_log": math.log1p(float(row["customer_tenure_days"])),
            "policy_tenure_log": math.log1p(float(row["policy_tenure_days"])),
            "verified_location": float(row["verified_location"]),
            "damage_consistency": float(row["damage_consistency"]),
            "document_verification_score": float(row["document_verification_score"]),
            "document_mismatch": float(row["document_mismatch"]),
            "duplicate_media": float(row["duplicate_media"]),
            "provider_prior_claims_log": math.log1p(count("provider")),
            "provider_prior_fraud_rate": rate("provider"),
            "surveyor_prior_claims_log": math.log1p(count("surveyor")),
            "surveyor_prior_fraud_rate": rate("surveyor"),
            "intermediary_prior_claims_log": math.log1p(count("intermediary")),
            "intermediary_prior_fraud_rate": rate("intermediary"),
            "bank_prior_claims_log": math.log1p(count("bank")),
            "device_prior_claims_log": math.log1p(count("device")),
            "phone_prior_claims_log": math.log1p(count("phone")),
            "address_prior_claims_log": math.log1p(count("address")),
            "customer_prior_claims_log": math.log1p(count("customer")),
            "customer_prior_fraud_rate": rate("customer"),
            "pair_prior_claims_log": math.log1p(pair_claim_counts[pair]),
            "pair_prior_fraud_rate": _smoothed_rate(pair_resolved_fraud[pair], pair_resolved_counts[pair]),
            "linked_prior_claims_log": math.log1p(len(linked_claim_ids)),
            "connected_known_fraud_log": math.log1p(len(connected_known)),
            "network_prior_fraud_rate": _smoothed_rate(linked_fraud, len(linked_resolved), prior=0.035, strength=12.0),
            "shared_entity_types": float(linked_entity_types),
            "relationship_reuse_score": float(relationship_reuse),
            "claim_type_motor": float(claim_type == "Motor"),
            "vehicle_segment_code": float(VEHICLE_SEGMENT_CODE.get(str(row["vehicle_segment"]), 0)),
            "connected_known_fraud_claims": connected_known,
        }
        feature_rows.append(feature)

        # Update information available immediately after FNOL.
        for kind, value in entities.items():
            key = (kind, value)
            entity_claim_counts[key] += 1
            entity_claim_ids[key].append(str(row["claim_id"]))
        pair_claim_counts[pair] += 1
        peer_stats[claim_type].add(amount)

        # Fraud outcome becomes available only after investigation completes.
        heappush(
            outcome_heap,
            (
                pd.Timestamp(row["outcome_date"]),
                str(row["claim_id"]),
                int(row["confirmed_fraud"]),
                entities,
                pair,
            ),
        )

    features = pd.DataFrame(feature_rows)
    derived_columns = [column for column in features.columns if column != "claim_id" and column not in df.columns]
    feature_only = features[derived_columns].reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), feature_only], axis=1)


class FraudMLEngine:
    """Time-safe hybrid fraud engine.

    Primary decision signal: calibrated supervised ML probability.
    Supporting signals: graph intelligence, anomaly detection and deterministic controls.
    Anomaly alone can escalate a claim only to review, not label it as fraud.
    """

    def __init__(self, claims: pd.DataFrame, random_state: int = 42):
        self.random_state = random_state
        self.claims = build_time_safe_features(claims)
        self.claims = self.claims.sort_values(["incident_date", "claim_id"]).reset_index(drop=True)
        self._prepare_splits()
        self._fit_supervised_model()
        self._fit_anomaly_model()
        self._score_components()
        self._calibrate_thresholds()
        self._assign_routes()
        self._compute_evaluation()
        self._compute_feature_importance()

    def _prepare_splits(self) -> None:
        n = len(self.claims)
        train_end = int(n * 0.65)
        val_end = int(n * 0.82)
        split = np.full(n, "test", dtype=object)
        split[:train_end] = "train"
        split[train_end:val_end] = "validation"
        self.claims["split"] = split
        self.train = self.claims[self.claims["split"] == "train"].copy()
        self.validation = self.claims[self.claims["split"] == "validation"].copy()
        self.test = self.claims[self.claims["split"] == "test"].copy()

    def _fit_supervised_model(self) -> None:
        x_train = self.train[MODEL_FEATURES].astype(float)
        y_train = self.train["confirmed_fraud"].astype(int)
        positive_weight = max((len(y_train) - y_train.sum()) / max(y_train.sum(), 1), 1.0)
        sample_weight = np.where(y_train.to_numpy() == 1, min(positive_weight, 8.0), 1.0)

        self.supervised_model = HistGradientBoostingClassifier(
            learning_rate=0.055,
            max_iter=280,
            max_leaf_nodes=17,
            min_samples_leaf=18,
            l2_regularization=0.7,
            random_state=self.random_state,
        )
        self.supervised_model.fit(x_train, y_train, sample_weight=sample_weight)

        val_raw = self.supervised_model.predict_proba(self.validation[MODEL_FEATURES].astype(float))[:, 1]
        val_logit = np.log(np.clip(val_raw, 1e-5, 1 - 1e-5) / np.clip(1 - val_raw, 1e-5, 1 - 1e-5))
        self.calibrator = LogisticRegression(random_state=self.random_state)
        self.calibrator.fit(val_logit.reshape(-1, 1), self.validation["confirmed_fraud"].astype(int))

        self.train_medians = self.train[MODEL_FEATURES].median().astype(float)

    def _calibrated_probability(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.supervised_model.predict_proba(frame[MODEL_FEATURES].astype(float))[:, 1]
        logit = np.log(np.clip(raw, 1e-5, 1 - 1e-5) / np.clip(1 - raw, 1e-5, 1 - 1e-5))
        return self.calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]

    def _fit_anomaly_model(self) -> None:
        self.anomaly_scaler = RobustScaler()
        x_train = self.anomaly_scaler.fit_transform(self.train[ANOMALY_FEATURES].astype(float))
        contamination = float(np.clip(self.train["confirmed_fraud"].mean() * 0.9, 0.025, 0.12))
        self.anomaly_model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=self.random_state,
        )
        self.anomaly_model.fit(x_train)
        train_raw = -self.anomaly_model.decision_function(x_train)
        self.train_anomaly_sorted = np.sort(train_raw)

    def _anomaly_percentiles(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.anomaly_scaler.transform(frame[ANOMALY_FEATURES].astype(float))
        raw = -self.anomaly_model.decision_function(x)
        ranks = np.searchsorted(self.train_anomaly_sorted, raw, side="right")
        return ranks / max(len(self.train_anomaly_sorted), 1)

    @staticmethod
    def _rule_score_row(row: pd.Series) -> float:
        score = 0.0
        score += 24.0 * float(row["document_mismatch"])
        score += 24.0 * float(row["duplicate_media"])
        score += max(0.0, 0.55 - float(row["document_verification_score"])) * 36.0
        score += max(0.0, 0.50 - float(row["damage_consistency"])) * 30.0
        score += min(float(row["connected_known_fraud_log"]) * 17.0, 30.0)
        score += min(max(float(row["claim_to_sum_insured"]) - 0.55, 0.0) * 28.0, 14.0)
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _graph_risk_row(row: pd.Series) -> float:
        score = (
            float(row["connected_known_fraud_log"]) * 23.0
            + float(row["network_prior_fraud_rate"]) * 62.0
            + float(row["pair_prior_fraud_rate"]) * 42.0
            + min(float(row["relationship_reuse_score"]) * 3.6, 24.0)
            + min(float(row["shared_entity_types"]) * 3.0, 18.0)
        )
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _legitimacy_score_row(row: pd.Series) -> float:
        score = (
            float(row["document_verification_score"]) * 27.0
            + float(row["damage_consistency"]) * 26.0
            + float(row["verified_location"]) * 12.0
            + min(float(row["customer_tenure_log"]) / 8.5, 1.0) * 12.0
            + min(float(row["policy_tenure_log"]) / 7.5, 1.0) * 9.0
            + float(row["catastrophe_event"]) * 9.0
            + (1.0 - min(float(row["claim_to_sum_insured"]), 1.0)) * 5.0
        )
        return float(np.clip(score, 0, 100))

    def _score_components(self) -> None:
        self.claims["fraud_probability"] = self._calibrated_probability(self.claims)
        self.claims["anomaly_percentile"] = self._anomaly_percentiles(self.claims)
        self.claims["rule_score"] = self.claims.apply(self._rule_score_row, axis=1)
        self.claims["graph_risk_score"] = self.claims.apply(self._graph_risk_row, axis=1)
        self.claims["legitimacy_score"] = self.claims.apply(self._legitimacy_score_row, axis=1)

    def _calibrate_thresholds(self) -> None:
        val = self.claims[self.claims["split"] == "validation"]
        y = val["confirmed_fraud"].to_numpy(dtype=int)
        p = val["fraud_probability"].to_numpy(dtype=float)
        precision, recall, thresholds = precision_recall_curve(y, p)

        if len(thresholds) == 0:
            self.red_threshold = 0.70
            self.amber_threshold = 0.30
            return

        # Red threshold: maximise recall while maintaining credible precision.
        candidates = [
            (thresholds[i], precision[i], recall[i])
            for i in range(len(thresholds))
            if precision[i] >= 0.68
        ]
        if candidates:
            self.red_threshold = float(sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)[0][0])
        else:
            f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
            self.red_threshold = float(thresholds[int(np.nanargmax(f1_values))])

        # Amber threshold: high-recall review gate. Anomaly alone never creates a Red verdict.
        candidates = [
            (thresholds[i], precision[i], recall[i])
            for i in range(len(thresholds))
            if recall[i] >= 0.88 and thresholds[i] < self.red_threshold
        ]
        if candidates:
            self.amber_threshold = float(sorted(candidates, key=lambda x: (x[1], x[0]), reverse=True)[0][0])
        else:
            self.amber_threshold = max(0.08, self.red_threshold * 0.42)

        self.red_threshold = float(np.clip(self.red_threshold, 0.35, 0.92))
        # A business floor prevents a tiny calibrated probability from flooding the
        # review queue. Network and anomaly gates below provide separate coverage
        # for novel patterns. The value would be tuned to Bajaj's SIU capacity.
        amber_floor = min(0.30, self.red_threshold - 0.08)
        self.amber_threshold = float(np.clip(max(self.amber_threshold, amber_floor), 0.06, self.red_threshold - 0.04))

    @staticmethod
    def _hard_override(row: pd.Series) -> bool:
        return bool(
            (int(row["document_mismatch"]) and int(row["duplicate_media"]))
            or (
                float(row["document_verification_score"]) < 0.22
                and float(row["damage_consistency"]) < 0.28
            )
        )

    def _assign_routes(self) -> None:
        routes: list[str] = []
        reasons: list[str] = []
        bands: list[str] = []
        overrides: list[bool] = []

        for _, row in self.claims.iterrows():
            p = float(row["fraud_probability"])
            anomaly = float(row["anomaly_percentile"])
            graph = float(row["graph_risk_score"])
            override = self._hard_override(row)
            overrides.append(override)

            if override or p >= self.red_threshold:
                routes.append("Red")
                bands.append("High")
                reasons.append(
                    "High calibrated fraud probability or a deterministic evidence conflict; pause straight-through settlement."
                )
            elif (
                p >= self.amber_threshold
                or (anomaly >= 0.98 and float(row["legitimacy_score"]) < 78.0)
                or (graph >= 80.0 and float(row["legitimacy_score"]) < 80.0)
            ):
                routes.append("Amber")
                bands.append("Medium")
                if anomaly >= 0.98 and p < self.amber_threshold:
                    reasons.append(
                        "Novel behaviour with insufficient legitimacy evidence. Request targeted evidence; anomaly alone is not proof of fraud."
                    )
                elif graph >= 80.0 and p < self.amber_threshold:
                    reasons.append("Elevated network risk and limited protective context require targeted human review.")
                else:
                    reasons.append("Moderate learned fraud probability; conduct targeted verification.")
            else:
                routes.append("Green")
                bands.append("Low")
                reasons.append("Low learned fraud probability with no critical anomaly or evidence conflict; fast-track with standard controls.")

        self.claims["route"] = routes
        self.claims["risk_band"] = bands
        self.claims["route_reason"] = reasons
        self.claims["hard_override"] = overrides
        self.claims["risk_score"] = (self.claims["fraud_probability"] * 100).round(1)

    @staticmethod
    def _classification_metrics(frame: pd.DataFrame, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
        y = frame["confirmed_fraud"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        genuine = max(tn + fp, 1)
        fraud_value_total = float(frame.loc[frame["confirmed_fraud"] == 1, "claim_amount"].sum())
        fraud_value_captured = float(frame.loc[(frame["confirmed_fraud"] == 1) & (prediction == 1), "claim_amount"].sum())
        return {
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "pr_auc": float(average_precision_score(y, probability)),
            "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else 0.0,
            "false_positive_rate": float(fp / genuine),
            "fraud_value_capture": float(fraud_value_captured / max(fraud_value_total, 1.0)),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    def _compute_evaluation(self) -> None:
        test = self.claims[self.claims["split"] == "test"].copy()
        red_prediction = ((test["fraud_probability"] >= self.red_threshold) | test["hard_override"]).astype(int).to_numpy()
        self.model_metrics = self._classification_metrics(
            test,
            red_prediction,
            test["fraud_probability"].to_numpy(dtype=float),
        )

        rules_prediction = (test["rule_score"] >= 42.0).astype(int).to_numpy()
        rule_probability = np.clip(test["rule_score"].to_numpy(dtype=float) / 100.0, 0, 1)
        self.rules_metrics = self._classification_metrics(test, rules_prediction, rule_probability)

        review_prediction = test["route"].isin(["Amber", "Red"]).astype(int).to_numpy()
        review_metrics = self._classification_metrics(
            test,
            review_prediction,
            test["fraud_probability"].to_numpy(dtype=float),
        )
        genuine_test = test[test["confirmed_fraud"] == 0]
        unusual_genuine = genuine_test[genuine_test["fraud_pattern"] == "UNUSUAL_GENUINE"]
        self.operational_metrics = {
            "review_precision": review_metrics["precision"],
            "review_recall": review_metrics["recall"],
            "fraud_value_capture": review_metrics["fraud_value_capture"],
            "genuine_fast_track_rate": float((genuine_test["route"] == "Green").mean()) if len(genuine_test) else 0.0,
            "unusual_genuine_fast_track_rate": float((unusual_genuine["route"] == "Green").mean()) if len(unusual_genuine) else 0.0,
            "review_rate": float(test["route"].isin(["Amber", "Red"]).mean()),
        }

        self.test_predictions = test
        self.confusion = np.array(
            [
                [self.model_metrics["tn"], self.model_metrics["fp"]],
                [self.model_metrics["fn"], self.model_metrics["tp"]],
            ]
        )
        precision, recall, thresholds = precision_recall_curve(
            test["confirmed_fraud"].astype(int), test["fraud_probability"].astype(float)
        )
        self.pr_curve = pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": np.append(thresholds, np.nan),
            }
        )

        prob_true, prob_pred = calibration_curve(
            test["confirmed_fraud"].astype(int),
            test["fraud_probability"].astype(float),
            n_bins=7,
            strategy="quantile",
        )
        self.calibration_data = pd.DataFrame({"predicted": prob_pred, "observed": prob_true})

    def _compute_feature_importance(self) -> None:
        test = self.claims[self.claims["split"] == "test"]
        x_test = test[MODEL_FEATURES].astype(float)
        y_test = test["confirmed_fraud"].astype(int)
        try:
            result = permutation_importance(
                self.supervised_model,
                x_test,
                y_test,
                n_repeats=5,
                random_state=self.random_state,
                scoring="average_precision",
            )
            importance = pd.DataFrame(
                {
                    "feature": MODEL_FEATURES,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            )
            importance["label"] = importance["feature"].map(FEATURE_LABELS)
            self.feature_importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            self.feature_importance = pd.DataFrame(
                {"feature": MODEL_FEATURES, "importance": 0.0, "std": 0.0, "label": [FEATURE_LABELS[f] for f in MODEL_FEATURES]}
            )

    def _probability_for_feature_row(self, values: pd.Series) -> float:
        frame = pd.DataFrame([[float(values[f]) for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
        return float(self._calibrated_probability(frame)[0])

    def explain_claim(self, claim_id: str, top_n: int = 6) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        match = self.claims[self.claims["claim_id"] == claim_id]
        if match.empty:
            raise KeyError(f"Claim {claim_id!r} not found")
        row = match.iloc[0]
        base_probability = float(row["fraud_probability"])
        contributions: list[dict[str, Any]] = []
        values = row[MODEL_FEATURES].astype(float).copy()

        for feature in MODEL_FEATURES:
            changed = values.copy()
            changed[feature] = float(self.train_medians[feature])
            changed_probability = self._probability_for_feature_row(changed)
            contribution = base_probability - changed_probability
            contributions.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS[feature],
                    "contribution": float(contribution),
                    "value": float(row[feature]),
                }
            )

        positive = sorted(
            [x for x in contributions if x["contribution"] > 0.002],
            key=lambda x: x["contribution"],
            reverse=True,
        )[:top_n]
        protective = sorted(
            [x for x in contributions if x["contribution"] < -0.002],
            key=lambda x: x["contribution"],
        )[:top_n]

        # Add explicit protective context even when model perturbation is diffuse.
        explicit = []
        if float(row["document_verification_score"]) >= 0.92:
            explicit.append({"feature": "document_verification_score", "label": "Strong document verification", "contribution": -0.03, "value": float(row["document_verification_score"])})
        if float(row["damage_consistency"]) >= 0.90:
            explicit.append({"feature": "damage_consistency", "label": "Damage supports the incident narrative", "contribution": -0.03, "value": float(row["damage_consistency"])})
        if int(row["catastrophe_event"]) == 1:
            explicit.append({"feature": "catastrophe_event", "label": "Known catastrophe explains the unusual cluster", "contribution": -0.025, "value": 1.0})
        if int(row["verified_location"]) == 1:
            explicit.append({"feature": "verified_location", "label": "Incident location verified", "contribution": -0.02, "value": 1.0})

        seen = {x["label"] for x in protective}
        for item in explicit:
            if item["label"] not in seen:
                protective.append(item)
                seen.add(item["label"])
        protective = sorted(protective, key=lambda x: x["contribution"])[:top_n]
        return positive, protective

    def score_claim(self, claim_id: str) -> RiskResult:
        match = self.claims[self.claims["claim_id"] == claim_id]
        if match.empty:
            raise KeyError(f"Claim {claim_id!r} not found")
        row = match.iloc[0]
        positive, protective = self.explain_claim(claim_id)
        return RiskResult(
            claim_id=claim_id,
            fraud_probability=round(float(row["fraud_probability"]), 4),
            risk_score=round(float(row["risk_score"]), 1),
            risk_band=str(row["risk_band"]),
            route=str(row["route"]),
            route_reason=str(row["route_reason"]),
            anomaly_percentile=round(float(row["anomaly_percentile"]), 4),
            graph_risk_score=round(float(row["graph_risk_score"]), 1),
            rule_score=round(float(row["rule_score"]), 1),
            legitimacy_score=round(float(row["legitimacy_score"]), 1),
            positive_reasons=positive,
            protective_reasons=protective,
            connected_known_fraud_claims=list(row["connected_known_fraud_claims"]),
            hard_override=bool(row["hard_override"]),
        )

    def comparison_table(self) -> pd.DataFrame:
        rows = []
        for name, metrics in [("Rules only", self.rules_metrics), ("ML + graph + controls", self.model_metrics)]:
            rows.append(
                {
                    "Approach": name,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "PR-AUC": metrics["pr_auc"],
                    "False-positive rate": metrics["false_positive_rate"],
                    "Fraud value captured": metrics["fraud_value_capture"],
                }
            )
        return pd.DataFrame(rows)

    def split_summary(self) -> pd.DataFrame:
        return (
            self.claims.groupby("split")
            .agg(
                claims=("claim_id", "count"),
                fraud_cases=("confirmed_fraud", "sum"),
                start_date=("incident_date", "min"),
                end_date=("incident_date", "max"),
            )
            .reset_index()
        )
