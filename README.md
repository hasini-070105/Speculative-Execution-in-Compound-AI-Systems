# Speculative Execution in Compound AI Systems
**Evaluating Query Routing Latency and Decision Quality Using Multi-Primitive Calibrated Discriminative Models**

**Author:** Taticherla Hasini  
**Department of Computer Science and Engineering, IIITDM Kurnool**

---

## Overview

Modern AI pipelines operate as multi-stage compound systems where an incoming user query passes through sequential control stages—intent classification, policy checking, and urgency scoring—before any downstream action or response is produced. Using large autoregressive language models (LLMs) at each intermediate control step creates high evaluation latency and unnecessary token overhead.

This project implements a **Speculative Jev Gateway** inspired by Kahneman's dual-process **System-1 / System-2** framework:
- **System-1 Calibrated Gate (`DeBERTa-v3-small`):** Positioned at the entry point of the pipeline, it evaluates all control checks in a single batched forward pass using three strongly typed decision primitives modeled after TypeSafe AI's Jev primitives (`Choice`, `Noul`, and `Score`) with temperature scaling.
- **System-2 Escalation:** Invoked only when queries require multi-sentence reasoning, fall outside the core domain, or carry high operational urgency.

---

## Typed Decision Primitives & Routing Architecture

In one batched tensor pass, `CalibratedSystem1Gate` scores the query against natural-language hypotheses representing three structured primitives:

1. **`Choice` (Domain Categorization):** Classifies the query across *App Support*, *Card Issue*, *Billing & Fees*, *Funds Transfer*, *Account Security*, and *Out-of-Domain / Complex*.
2. **`Noul` (Generative Necessity):** Binary predicate predicting whether the query needs multi-sentence explanation (`True`) or can be satisfied by a direct deterministic action (`False`).
3. **`Score` (Ordinal Urgency Rubric):** Scores operational risk across ordered levels (*Routine*, *Moderate*, *High Urgency*, and *Critical Fraud*) and computes expected urgency.

Using calibrated joint routing confidence, the gateway dispatches queries along three paths:

- **Path A — Deterministic Bypass:** High-confidence routine commands execute deterministic backend actions directly without invoking any generative LLM.
- **Path B — Fast Single-Step Handling:** High-confidence domain questions requiring explanation skip intermediate routing LLMs and proceed directly to lightweight single-step synthesis.
- **Path C — Calibrated Deep Escalation:** Low-confidence, out-of-domain, or high-urgency queries escalate safely to System-2 reasoning.

---

## Evaluation Methodology

The benchmark framework evaluates the **Sequential LLM Chain (Baseline)** against the **Speculative Jev Gateway** across four query categories (*Direct Action Commands*, *Domain Policy Inquiries*, *High-Urgency Security Alerts*, and *Out-of-Domain / Ambiguous Queries*) using four core metrics:
- **Gate & End-to-End Evaluation Time:** Wall-clock latency from receiving the query to finalizing the routing decision and downstream dispatch.
- **Routing Accuracy:** Alignment between predicted paths and ground-truth optimal execution paths.
- **Escalation Recall on Risky/OOD Queries:** Ability to catch high-urgency or out-of-domain inputs without false automation.
- **Expected Calibration Error (ECE):** Alignment between predicted joint confidence and empirical routing accuracy.

---

## Repository Structure

```text
speculative-jev-gateway/
├── README.md                  # Project documentation and architecture overview
├── requirements.txt           # Python dependencies
├── .gitignore                 # Standard Python ignores
│
├── jev_primitives.py          # Choice, Noul, Score primitives & CalibratedSystem1Gate
├── speculative_gateway.py     # Three-way calibrated routing & Baseline vs. Jev engines
├── evaluate.py                # Async benchmark runner & monochrome chart exporter
│
├── docs/
│   ├── technical_report.pdf   # Full technical report
│   └── presentation.pptx      # Project presentation slides
│
└── output/                    # Exported single-channel ("L" mode) B&W PNG comparison plots
