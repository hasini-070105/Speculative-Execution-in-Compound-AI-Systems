# ==============================================================================
# Speculative Execution in Compound AI Systems:
# Deterministic Path A Handlers & Dual Execution Pipelines (Baseline vs. Jev)
# ==============================================================================

import asyncio
import time
from typing import Optional, Tuple

from jev_primitives import (
    CalibratedSystem1Gate,
    DecisionResult,
    ExecutionTelemetry,
    GroundedLocalSystem2Service,
    RoutingPath,
)


# ==============================================================================
# 1. Deterministic Action Templates (Path A Zero-Token Bypass)
# ==============================================================================

def get_domain_template_answer(query: str, mode: str = "direct") -> Optional[str]:
    """Handles deterministic actions & standard policy queries instantaneously with 0 LLM tokens."""
    ql = str(query).strip().lower()

    # 1. App Download / 2FA / Unsubscribe (Deterministic Path A)
    if any(k in ql for k in ["download", "playstore", "play store", "app store", "install", "apk", "mobile app", "where to get", "two-factor", "2fa", "unsubscribe"]):
        if "two-factor" in ql or "2fa" in ql:
            return "To enable Two-Factor Authentication (2FA), open Profile > Security > Enable 2FA and link your authenticator app or SMS verification."
        if "unsubscribe" in ql:
            return "Marketing email notifications have been disabled in your communication preferences."
        return "You can download the official mobile app from the Google Play Store (Android) or Apple App Store (iOS) by searching 'Apex Mobile' or visiting apexbank.com/download."

    # 2. Card Policy / Freeze / PIN (Deterministic Path A)
    if any(k in ql for k in ["freeze", "lock", "block", "misplaced", "lost", "card", "pin", "cvv", "replacement", "unblock"]):
        return "Your card freeze/security lock has been applied immediately, blocking all new purchases and ATM withdrawals."

    # 3. Billing / Fees / Plans / Pending Authorizations
    if any(k in ql for k in ["fee", "charge", "charged", "bill", "refund", "dispute", "statement", "overdraft", "plan", "tier", "cashback", "pending", "authorization", "promo", "balance"]):
        if "plan" in ql or "tier" in ql:
            return "The Basic tier ($0/month) includes free ACH transfers and standard debit access, while the Pro Monthly plan ($49.99/month) adds zero foreign exchange fees, priority support, and 2% cashback."
        if "cashback" in ql:
            return "Cashback rewards are calculated at 1.5% on eligible settled online retail transactions and credited automatically at the end of each billing cycle."
        if "pending" in ql or "authorization" in ql:
            return "Pending card authorizations are temporary holds placed by merchants and automatically clear or drop off within 3 to 5 business days once final settlement completes."
        return "Standard international transaction fees are 1.5% on the Basic tier and $0 on the Pro plan, with itemized PDF statements generated monthly."

    # 4. Transfers / ACH / Wire / Routing
    if any(k in ql for k in ["transfer", "wire", "send money", "routing", "swift", "iban", "ach", "zelle", "limit"]):
        if "limit" in ql and mode == "direct":
            return "Standard daily withdrawal and transfer limits are $5,000 for ATM/debit and $25,000 for domestic ACH/wire transfers."
        return "Standard ACH transfers take 1–3 business days with $0 fee, whereas domestic wire transfers settle same-day for a $15 fee (ABA Routing: 122000496, SWIFT: APEXUS33)."

    # 5. Security / Fraud / Phishing / Password
    if any(k in ql for k in ["fraud", "hacked", "stolen", "unauthorized", "suspicious", "breach", "locked out", "phishing", "password"]):
        if "password" in ql and mode == "direct":
            return "A secure password reset link has been dispatched to your registered email."
        return "Immediate security protocol engaged: Active sessions terminated and protective lock applied while our 24/7 Fraud Unit reviews activity."

    return None


# ==============================================================================
# 2. Dual Execution Pipelines: Without Our Model vs. With Our Model
# ==============================================================================

class BaselineSequentialChain:
    """WITHOUT OUR MODEL: Traditional sequential generative LLM agent chain (Routing -> Policy Check -> Synthesis)."""

    def __init__(self, llm: GroundedLocalSystem2Service):
        self.llm = llm

    async def execute(self, query: str) -> ExecutionTelemetry:
        t0 = time.perf_counter()
        timeline = []

        # Step 1: Sequential LLM Intent & Routing
        s1_start = time.perf_counter() - t0
        _, tok1, lat1 = await self.llm.run_baseline_step1_intent(query)
        s1_end = time.perf_counter() - t0
        timeline.append(("Step 1: LLM Routing", s1_start, s1_end))

        # Step 2: Sequential LLM Policy Verification
        s2_start = time.perf_counter() - t0
        _, tok2, lat2 = await self.llm.run_baseline_step2_policy(query)
        s2_end = time.perf_counter() - t0
        timeline.append(("Step 2: LLM Policy Check", s2_start, s2_end))

        # Step 3: Sequential LLM Prose Synthesis
        s3_start = time.perf_counter() - t0
        reply, tok3, lat3 = await self.llm.run_baseline_step3_synthesis(query)
        s3_end = time.perf_counter() - t0
        timeline.append(("Step 3: LLM Synthesis", s3_start, s3_end))

        total_sec = time.perf_counter() - t0
        total_tokens = tok1 + tok2 + tok3
        cost_usd = round(total_tokens * 0.00003, 5)

        why = (
            f"Traditional agent chains invoke autoregressive LLMs sequentially for routing, policy checks, and prose generation. "
            f"This consumes {total_tokens} tokens and incurs {total_sec:.2f}s of cumulative latency."
        )

        return ExecutionTelemetry(
            query=query,
            mode="Without Our Model (Sequential LLM Chain)",
            path_taken="Sequential Multi-Turn LLM Chain",
            gate_latency_sec=0.0,
            llm_latency_sec=round(lat1 + lat2 + lat3, 3),
            total_latency_sec=round(total_sec, 3),
            tokens_consumed=total_tokens,
            cost_usd=cost_usd,
            output_text=reply,
            why_happened=why,
            timeline_events=timeline,
        )


class SpeculativeJevGateway:
    """WITH OUR MODEL: Multi-Primitive Calibrated Jev Gate + Speculative Dispatch."""

    def __init__(self, gate: CalibratedSystem1Gate, llm: GroundedLocalSystem2Service):
        self.gate = gate
        self.llm = llm

    async def execute(self, query: str, tau_override: Optional[float] = None) -> Tuple[ExecutionTelemetry, DecisionResult]:
        t0 = time.perf_counter()
        timeline = []

        # Stage 1: Fast System-1 Jev Decision Gate
        g_start = time.perf_counter() - t0
        decision = await self.gate.evaluate(query, tau_override=tau_override)
        g_end = time.perf_counter() - t0
        timeline.append(("Jev System-1 Gate", g_start, g_end))

        total_tokens = 0
        cost_usd = 0.0
        llm_lat = 0.0

        if decision.routing_path == RoutingPath.PATH_A:
            # PATH A: Deterministic Bypass — Skips the LLM completely
            t_act_start = time.perf_counter() - t0
            await asyncio.sleep(0.035)
            t_act_end = time.perf_counter() - t0
            timeline.append(("Direct Backend Action", t_act_start, t_act_end))

            reply = get_domain_template_answer(query, mode="direct") or "Action completed."
            total_tokens = 0
            cost_usd = 0.00002

        elif decision.routing_path == RoutingPath.PATH_B:
            # PATH B: Direct Fast LLM Synthesis
            llm_start = time.perf_counter() - t0
            reply, tokens, llm_lat = await self.llm.run_speculative_light_synthesis(
                query=query,
                category=decision.category.choice,
            )
            llm_end = time.perf_counter() - t0
            timeline.append(("Fast LLM Synthesis", llm_start, llm_end))
            total_tokens = tokens
            cost_usd = round(0.00002 + tokens * 0.0000015, 5)

        else:
            # PATH C: Calibrated Escalation to Web-Grounded System-2 LLM
            llm_start = time.perf_counter() - t0
            reply, tokens, llm_lat = await self.llm.run_deep_escalation(
                query=query,
                category=decision.category.choice,
            )
            llm_end = time.perf_counter() - t0
            timeline.append(("Web-Grounded System-2 LLM", llm_start, llm_end))
            total_tokens = tokens
            cost_usd = round(0.00002 + tokens * 0.00003, 5)

        total_sec = time.perf_counter() - t0

        telemetry = ExecutionTelemetry(
            query=query,
            mode="With Our Model (Speculative Jev Gateway)",
            path_taken=decision.routing_path.value,
            gate_latency_sec=decision.gate_latency_sec,
            llm_latency_sec=round(llm_lat, 3),
            total_latency_sec=round(total_sec, 3),
            tokens_consumed=total_tokens,
            cost_usd=cost_usd,
            output_text=reply,
            why_happened=decision.routing_reason,
            timeline_events=timeline,
        )
        return telemetry, decision
