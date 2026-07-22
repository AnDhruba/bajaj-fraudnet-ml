# FraudNet ML: Points to Carry into the Final PPT

## Core claim we can honestly make about the upgraded prototype

The demonstrator now combines:

- supervised behavioural machine learning;
- time-safe graph features;
- unsupervised anomaly detection;
- deterministic evidence controls;
- legitimacy signals that reduce false positives;
- calibrated Green / Amber / Red routing;
- chronological train, validation and test evaluation.

The model does not equate unusual behaviour with fraud. Anomaly alone can request targeted verification but cannot independently create a Red fraud verdict.

## Why this is better than fixed rules

Fixed thresholds are predictable and can be evaded. The supervised model learns non-linear interactions, such as ordinary claim values becoming suspicious only when combined with repeated provider-surveyor behaviour, shared infrastructure and weak supporting evidence.

At the same time, unusual genuine claims can remain Green when the model sees strong legitimacy context: insured value, incident severity, verified location, coherent damage, trusted documents, catastrophe context and customer tenure.

## Current prototype limitations to state transparently

- Synthetic proxy data, not Bajaj claims
- Exact identifiers rather than production-grade probabilistic entity resolution
- Limited claim products and ecosystem features
- Small graph unsuitable for claiming production GNN performance
- Synthetic performance metrics are methodology demonstrations, not business forecasts

## Next improvements

1. Train and validate on de-identified Bajaj claims and SIU outcomes.
2. Add probabilistic entity resolution for aliases and rotated identities.
3. Optimise thresholds to SIU capacity and fraud rupee-value capture.
4. Add investigator feedback, appeal outcomes and recovery results.
5. Monitor data drift, prediction drift and segment performance.
6. Add graph embeddings / GraphSAGE / link prediction after enough labelled graph history exists.
7. Add role-based access, audit trails, adverse-decision controls and model-risk governance.

## Metrics for the PPT

- Precision
- Recall
- PR-AUC
- False-positive rate
- Fraud value captured
- Genuine-claim fast-track rate
- Unusual-genuine fast-track rate
- Recall at fixed SIU review capacity
- Lift over the existing rule engine
- Time from FNOL to risk decision

## Recommended sentence

“FraudNet does not ask whether a claim is merely unusual. It learns whether the claim’s behaviour, context and ecosystem relationships resemble confirmed fraud, while treating verified legitimacy as evidence in the customer’s favour.”
