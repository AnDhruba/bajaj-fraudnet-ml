from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 42
    as_of_date: date = date(2026, 7, 15)
    n_regular_claims: int = 1_150
    n_motor_ring_claims: int = 54
    n_health_ring_claims: int = 46
    n_subtle_fraud_claims: int = 42
    n_unusual_genuine_claims: int = 64


CITIES = [
    "Mumbai", "Pune", "Nagpur", "Nashik", "Jalgaon", "Thane", "Chhatrapati Sambhajinagar",
    "Kolhapur", "Surat", "Indore", "Bhopal", "Delhi"
]

VEHICLE_SEGMENTS = {
    "Economy": (550_000, 0.10),
    "Mid": (1_100_000, 0.16),
    "Premium": (2_600_000, 0.22),
    "Luxury": (7_500_000, 0.30),
}


def _choice(rng: random.Random, items: list[str]) -> str:
    return items[rng.randrange(len(items))]


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def generate_claims(config: DatasetConfig = DatasetConfig()) -> pd.DataFrame:
    """Create a synthetic insurance claims history for a fraud-analytics demonstrator.

    The dataset contains:
    - ordinary genuine claims;
    - opportunistic fraud;
    - two coordinated ecosystem fraud rings;
    - subtle fraud designed to evade simple thresholds;
    - unusual but genuine claims designed to test false-positive control.

    No row is intended to represent a real person or real Bajaj customer.
    """
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)
    today = config.as_of_date

    garages = [f"GAR-{i:03d}" for i in range(1, 41)]
    hospitals = [f"HOS-{i:03d}" for i in range(1, 28)]
    surveyors = [f"SUR-{i:03d}" for i in range(1, 34)]
    tpas = [f"TPA-{i:03d}" for i in range(1, 11)]
    customers = [f"CUS-{i:05d}" for i in range(1, 980)]
    policies = [f"POL-{i:06d}" for i in range(1, 1_850)]
    banks = [f"BANK-{i:06d}" for i in range(1, 1_050)]
    phones = [f"+91-9{rng.randrange(100000000, 999999999)}" for _ in range(1_100)]
    devices = [f"DEV-{i:05d}" for i in range(1, 930)]
    addresses = [f"ADDR-{i:05d}" for i in range(1, 1_000)]

    rows: list[dict] = []
    claim_counter = 1

    def add_claim(
        *,
        claim_type: str,
        customer_id: str,
        provider_id: str,
        provider_type: str,
        intermediary_id: str,
        intermediary_type: str,
        surveyor_id: str,
        bank_account: str,
        phone: str,
        address: str,
        device_id: str,
        city: str,
        claim_amount: float,
        sum_insured: float,
        incident_date: date,
        incident_hour: int,
        vehicle_segment: str,
        incident_severity: int,
        catastrophe_event: int,
        customer_tenure_days: int,
        policy_tenure_days: int,
        verified_location: int,
        damage_consistency: float,
        document_verification_score: float,
        document_mismatch: int,
        duplicate_media: int,
        confirmed_fraud: int,
        fraud_pattern: str,
        case_note: str,
        resolution_days: int,
    ) -> None:
        nonlocal claim_counter
        claim_id = f"CLM-{claim_counter:06d}"
        outcome_date = incident_date + timedelta(days=resolution_days)
        rows.append(
            {
                "claim_id": claim_id,
                "policy_id": _choice(rng, policies),
                "customer_id": customer_id,
                "claim_type": claim_type,
                "claim_amount": round(float(claim_amount), 2),
                "sum_insured": round(float(sum_insured), 2),
                "incident_date": incident_date.isoformat(),
                "outcome_date": outcome_date.isoformat(),
                "incident_hour": int(incident_hour),
                "provider_id": provider_id,
                "provider_type": provider_type,
                "intermediary_id": intermediary_id,
                "intermediary_type": intermediary_type,
                "surveyor_id": surveyor_id,
                "bank_account": bank_account,
                "phone": phone,
                "address": address,
                "device_id": device_id,
                "city": city,
                "vehicle_segment": vehicle_segment,
                "incident_severity": int(incident_severity),
                "catastrophe_event": int(catastrophe_event),
                "customer_tenure_days": int(customer_tenure_days),
                "policy_tenure_days": int(policy_tenure_days),
                "verified_location": int(verified_location),
                "damage_consistency": round(float(damage_consistency), 3),
                "document_verification_score": round(float(document_verification_score), 3),
                "document_mismatch": int(document_mismatch),
                "duplicate_media": int(duplicate_media),
                "confirmed_fraud": int(confirmed_fraud),
                "fraud_pattern": fraud_pattern,
                "case_note": case_note,
                "resolution_days": int(resolution_days),
                "claim_status": "Rejected - Fraud" if confirmed_fraud else _choice(
                    rng, ["Settled", "Settled", "Settled", "Approved", "Closed"]
                ),
            }
        )
        claim_counter += 1

    # -----------------------------
    # Ordinary claims and opportunistic fraud
    # -----------------------------
    for _ in range(config.n_regular_claims):
        incident_date = today - timedelta(days=rng.randrange(8, 920))
        claim_type = "Motor" if rng.random() < 0.62 else "Health"
        customer = _choice(rng, customers)
        customer_idx = int(customer.split("-")[1]) - 1
        city = _choice(rng, CITIES)
        customer_tenure = rng.randrange(120, 4_200)
        policy_tenure = rng.randrange(20, min(customer_tenure, 1_600))
        verified_location = int(rng.random() < 0.93)
        catastrophe = int(rng.random() < 0.025)
        doc_score = _clamp(np_rng.normal(0.91, 0.065), 0.50, 0.999)
        damage_consistency = _clamp(np_rng.normal(0.88, 0.09), 0.30, 0.999)
        incident_hour = rng.randrange(0, 24)
        severity = int(np_rng.choice([1, 2, 3, 4, 5], p=[0.17, 0.30, 0.29, 0.18, 0.06]))

        if claim_type == "Motor":
            segment = rng.choices(
                population=list(VEHICLE_SEGMENTS),
                weights=[0.48, 0.34, 0.14, 0.04],
                k=1,
            )[0]
            base_sum, severity_ratio = VEHICLE_SEGMENTS[segment]
            sum_insured = base_sum * rng.uniform(0.72, 1.35)
            provider = _choice(rng, garages)
            surveyor = _choice(rng, surveyors)
            intermediary = "NA"
            intermediary_type = "None"
            amount = sum_insured * severity_ratio * severity / 3.0 * rng.uniform(0.55, 1.25)
            amount = _clamp(amount, 7_000, sum_insured * 0.72)
            provider_type = "Garage"
        else:
            segment = "NA"
            sum_insured = rng.choice([300_000, 500_000, 750_000, 1_000_000, 1_500_000])
            provider = _choice(rng, hospitals)
            surveyor = "NA"
            intermediary = _choice(rng, tpas)
            intermediary_type = "TPA"
            amount = np_rng.lognormal(mean=10.7, sigma=0.58) * (0.70 + severity * 0.16)
            amount = _clamp(amount, 8_000, sum_insured * 0.85)
            provider_type = "Hospital"

        opportunistic = int(rng.random() < 0.034)
        mismatch = 0
        duplicate = 0
        fraud_pattern = "NONE"
        case_note = "Ordinary claim with context consistent with the reported incident."
        if opportunistic:
            fraud_pattern = "OPPORTUNISTIC"
            amount = _clamp(amount * rng.uniform(1.18, 1.75), 8_000, sum_insured * 0.96)
            mismatch = int(rng.random() < 0.52)
            duplicate = int(rng.random() < 0.30)
            doc_score = _clamp(doc_score - rng.uniform(0.18, 0.42), 0.08, 0.92)
            damage_consistency = _clamp(damage_consistency - rng.uniform(0.12, 0.38), 0.05, 0.92)
            verified_location = int(rng.random() < 0.45)
            case_note = "Opportunistic fraud with inflated loss or inconsistent supporting evidence."

        add_claim(
            claim_type=claim_type,
            customer_id=customer,
            provider_id=provider,
            provider_type=provider_type,
            intermediary_id=intermediary,
            intermediary_type=intermediary_type,
            surveyor_id=surveyor,
            bank_account=banks[customer_idx % len(banks)],
            phone=phones[customer_idx % len(phones)],
            address=addresses[customer_idx % len(addresses)],
            device_id=devices[customer_idx % len(devices)],
            city=city,
            claim_amount=amount,
            sum_insured=sum_insured,
            incident_date=incident_date,
            incident_hour=incident_hour,
            vehicle_segment=segment,
            incident_severity=severity,
            catastrophe_event=catastrophe,
            customer_tenure_days=customer_tenure,
            policy_tenure_days=policy_tenure,
            verified_location=verified_location,
            damage_consistency=damage_consistency,
            document_verification_score=doc_score,
            document_mismatch=mismatch,
            duplicate_media=duplicate,
            confirmed_fraud=opportunistic,
            fraud_pattern=fraud_pattern,
            case_note=case_note,
            resolution_days=rng.randrange(12, 75) if opportunistic else rng.randrange(4, 38),
        )

    # -----------------------------
    # Coordinated motor fraud ring
    # -----------------------------
    motor_customers = [f"RMC-{i:03d}" for i in range(1, 29)]
    motor_banks = ["BANK-RING-M1", "BANK-RING-M2", "BANK-RING-M3", "BANK-RING-M4"]
    motor_phones = ["+91-9999001001", "+91-9999001002", "+91-9999001003", "+91-9999001004"]
    motor_addresses = ["ADDR-RING-M1", "ADDR-RING-M2", "ADDR-RING-M3"]
    motor_devices = ["DEV-RING-M1", "DEV-RING-M2", "DEV-RING-M3"]
    for i in range(config.n_motor_ring_claims):
        incident_date = today - timedelta(days=880 - int(i * 15.2) + rng.randrange(-6, 7))
        severity = rng.choice([2, 3, 4])
        sum_insured = rng.uniform(900_000, 2_900_000)
        amount = sum_insured * rng.uniform(0.13, 0.28)
        confirmed = 1 if i < 45 else int(rng.random() < 0.78)
        add_claim(
            claim_type="Motor",
            customer_id=motor_customers[i % len(motor_customers)],
            provider_id="GAR-007",
            provider_type="Garage",
            intermediary_id="NA",
            intermediary_type="None",
            surveyor_id="SUR-004" if i < 43 else "SUR-009",
            bank_account=motor_banks[i % len(motor_banks)],
            phone=motor_phones[i % len(motor_phones)],
            address=motor_addresses[i % len(motor_addresses)],
            device_id=motor_devices[i % len(motor_devices)],
            city="Mumbai" if i % 3 else "Thane",
            claim_amount=amount,
            sum_insured=sum_insured,
            incident_date=incident_date,
            incident_hour=2 if i % 2 else 3,
            vehicle_segment=rng.choice(["Mid", "Premium"]),
            incident_severity=severity,
            catastrophe_event=0,
            customer_tenure_days=rng.randrange(90, 1_900),
            policy_tenure_days=rng.randrange(15, 560),
            verified_location=int(i % 4 != 0),
            damage_consistency=_clamp(np_rng.normal(0.55, 0.15), 0.10, 0.88),
            document_verification_score=_clamp(np_rng.normal(0.64, 0.16), 0.12, 0.93),
            document_mismatch=int(i % 5 == 0),
            duplicate_media=int(i % 4 == 0),
            confirmed_fraud=confirmed,
            fraud_pattern="MOTOR_COLLUSION_RING",
            case_note="Coordinated garage-surveyor ring using shared payment and submission infrastructure.",
            resolution_days=rng.randrange(24, 105),
        )

    # -----------------------------
    # Coordinated hospital/TPA fraud ring
    # -----------------------------
    health_customers = [f"RHC-{i:03d}" for i in range(1, 25)]
    health_banks = ["BANK-RING-H1", "BANK-RING-H2", "BANK-RING-H3"]
    health_phones = ["+91-9888002001", "+91-9888002002", "+91-9888002003"]
    health_addresses = ["ADDR-RING-H1", "ADDR-RING-H2", "ADDR-RING-H3"]
    health_devices = ["DEV-RING-H1", "DEV-RING-H2"]
    for i in range(config.n_health_ring_claims):
        incident_date = today - timedelta(days=835 - int(i * 17.0) + rng.randrange(-8, 9))
        sum_insured = rng.choice([500_000, 750_000, 1_000_000, 1_500_000])
        amount = sum_insured * rng.uniform(0.22, 0.48)
        confirmed = 1 if i < 38 else int(rng.random() < 0.76)
        add_claim(
            claim_type="Health",
            customer_id=health_customers[i % len(health_customers)],
            provider_id="HOS-005",
            provider_type="Hospital",
            intermediary_id="TPA-002",
            intermediary_type="TPA",
            surveyor_id="NA",
            bank_account=health_banks[i % len(health_banks)],
            phone=health_phones[i % len(health_phones)],
            address=health_addresses[i % len(health_addresses)],
            device_id=health_devices[i % len(health_devices)],
            city="Pune",
            claim_amount=amount,
            sum_insured=sum_insured,
            incident_date=incident_date,
            incident_hour=1 if i % 2 else 2,
            vehicle_segment="NA",
            incident_severity=rng.choice([2, 3, 4]),
            catastrophe_event=0,
            customer_tenure_days=rng.randrange(75, 1_600),
            policy_tenure_days=rng.randrange(12, 500),
            verified_location=int(i % 5 != 0),
            damage_consistency=_clamp(np_rng.normal(0.58, 0.16), 0.12, 0.91),
            document_verification_score=_clamp(np_rng.normal(0.67, 0.15), 0.15, 0.94),
            document_mismatch=int(i % 6 == 0),
            duplicate_media=int(i % 5 == 0),
            confirmed_fraud=confirmed,
            fraud_pattern="HEALTH_COLLUSION_RING",
            case_note="Coordinated hospital-TPA ring with repeated identities and abnormal billing relationships.",
            resolution_days=rng.randrange(28, 115),
        )

    # -----------------------------
    # Subtle fraud designed to evade simple thresholds
    # -----------------------------
    subtle_devices = ["DEV-SUBTLE-01", "DEV-SUBTLE-02", "DEV-SUBTLE-03", "DEV-SUBTLE-04"]
    for i in range(config.n_subtle_fraud_claims):
        incident_date = today - timedelta(days=740 - int(i * 16.8) + rng.randrange(-5, 6))
        customer = f"SFC-{i:04d}"
        claim_type = "Motor" if i % 3 else "Health"
        if claim_type == "Motor":
            segment = rng.choice(["Economy", "Mid", "Premium"])
            sum_insured = VEHICLE_SEGMENTS[segment][0] * rng.uniform(0.85, 1.15)
            amount = sum_insured * rng.uniform(0.08, 0.18)  # deliberately ordinary
            provider = "GAR-021"
            surveyor = "SUR-021" if i % 4 else "SUR-022"
            intermediary = "NA"
            intermediary_type = "None"
            provider_type = "Garage"
        else:
            segment = "NA"
            sum_insured = rng.choice([500_000, 750_000, 1_000_000])
            amount = sum_insured * rng.uniform(0.10, 0.22)
            provider = "HOS-019"
            surveyor = "NA"
            intermediary = "TPA-008"
            intermediary_type = "TPA"
            provider_type = "Hospital"

        add_claim(
            claim_type=claim_type,
            customer_id=customer,
            provider_id=provider,
            provider_type=provider_type,
            intermediary_id=intermediary,
            intermediary_type=intermediary_type,
            surveyor_id=surveyor,
            bank_account=f"BANK-SUB-{i:04d}",  # rotates direct identifiers
            phone=f"+91-9777{i:06d}",
            address=f"ADDR-SUB-{i:04d}",
            device_id=subtle_devices[i % len(subtle_devices)],
            city="Nashik" if i % 2 else "Nagpur",
            claim_amount=amount,
            sum_insured=sum_insured,
            incident_date=incident_date,
            incident_hour=rng.randrange(8, 21),
            vehicle_segment=segment,
            incident_severity=rng.choice([2, 3]),
            catastrophe_event=0,
            customer_tenure_days=rng.randrange(300, 2_300),
            policy_tenure_days=rng.randrange(120, 760),
            verified_location=1,
            damage_consistency=_clamp(np_rng.normal(0.82, 0.06), 0.64, 0.96),
            document_verification_score=_clamp(np_rng.normal(0.89, 0.045), 0.74, 0.98),
            document_mismatch=0,
            duplicate_media=0,
            confirmed_fraud=1,
            fraud_pattern="SUBTLE_BEHAVIOURAL_RING",
            case_note="Fraud deliberately kept below obvious thresholds; detected through recurring behavioural and network patterns.",
            resolution_days=rng.randrange(35, 120),
        )

    # -----------------------------
    # Unusual but genuine claims: explicit false-positive stress tests
    # -----------------------------
    for i in range(config.n_unusual_genuine_claims):
        incident_date = today - timedelta(days=rng.randrange(10, 760))
        scenario = i % 3

        if scenario == 0:  # genuine luxury loss
            claim_type = "Motor"
            segment = "Luxury"
            sum_insured = rng.uniform(5_500_000, 12_500_000)
            amount = sum_insured * rng.uniform(0.18, 0.38)
            provider = _choice(rng, ["GAR-031", "GAR-032", "GAR-033"])
            surveyor = _choice(rng, ["SUR-028", "SUR-029"])
            intermediary = "NA"
            intermediary_type = "None"
            provider_type = "Garage"
            city = _choice(rng, ["Mumbai", "Delhi", "Pune"])
            device = f"DEV-LUX-{i:03d}"
            catastrophe = 0
            case_note = "High-value luxury-vehicle loss supported by severity, insured value and verified evidence."
        elif scenario == 1:  # catastrophe cluster
            claim_type = "Motor"
            segment = rng.choice(["Economy", "Mid", "Premium"])
            sum_insured = VEHICLE_SEGMENTS[segment][0] * rng.uniform(0.8, 1.25)
            amount = sum_insured * rng.uniform(0.11, 0.27)
            provider = "GAR-036"
            surveyor = "SUR-031"
            intermediary = "NA"
            intermediary_type = "None"
            provider_type = "Garage"
            city = "Pune"
            device = "DEV-CATASTROPHE-PORTAL"  # shared but legitimately so
            catastrophe = 1
            case_note = "Genuine hailstorm cluster sharing the same garage, surveyor and catastrophe intake portal."
        else:  # employer-assisted health submissions
            claim_type = "Health"
            segment = "NA"
            sum_insured = rng.choice([500_000, 750_000, 1_000_000])
            amount = sum_insured * rng.uniform(0.17, 0.42)
            provider = "HOS-023"
            surveyor = "NA"
            intermediary = "TPA-010"
            intermediary_type = "TPA"
            provider_type = "Hospital"
            city = "Mumbai"
            device = "DEV-CORPORATE-HELPDESK"
            catastrophe = 0
            case_note = "Genuine employer-assisted health claim submitted through a shared corporate benefits desk."

        add_claim(
            claim_type=claim_type,
            customer_id=f"UGC-{i:04d}",
            provider_id=provider,
            provider_type=provider_type,
            intermediary_id=intermediary,
            intermediary_type=intermediary_type,
            surveyor_id=surveyor,
            bank_account=f"BANK-UG-{i:04d}",
            phone=f"+91-9666{i:06d}",
            address=f"ADDR-UG-{i:04d}",
            device_id=device,
            city=city,
            claim_amount=amount,
            sum_insured=sum_insured,
            incident_date=incident_date,
            incident_hour=rng.randrange(0, 24),
            vehicle_segment=segment,
            incident_severity=rng.choice([4, 5]),
            catastrophe_event=catastrophe,
            customer_tenure_days=rng.randrange(1_500, 5_000),
            policy_tenure_days=rng.randrange(420, 1_700),
            verified_location=1,
            damage_consistency=_clamp(np_rng.normal(0.96, 0.025), 0.86, 0.999),
            document_verification_score=_clamp(np_rng.normal(0.975, 0.012), 0.91, 0.999),
            document_mismatch=0,
            duplicate_media=0,
            confirmed_fraud=0,
            fraud_pattern="UNUSUAL_GENUINE",
            case_note=case_note,
            resolution_days=rng.randrange(5, 34),
        )

    df = pd.DataFrame(rows)
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["outcome_date"] = pd.to_datetime(df["outcome_date"])
    df = df.sort_values(["incident_date", "claim_id"]).reset_index(drop=True)
    return df


def save_dataset(output_path: str | Path, config: DatasetConfig = DatasetConfig()) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_claims(config)
    df.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    path = save_dataset(Path(__file__).resolve().parents[1] / "data" / "claims.csv")
    print(f"Generated synthetic claims dataset at {path}")
