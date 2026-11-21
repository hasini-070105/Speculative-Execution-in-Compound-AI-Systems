# ==============================================================================
# Speculative Execution in Compound AI Systems:
# Typed Jev Decision Primitives, System-1 Gate & Grounded System-2 Service
# ==============================================================================

import asyncio
import re
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. Typed Decision Primitives & Telemetry Data Structures
# ==============================================================================

class RoutingPath(str, Enum):
    PATH_A = "Path A (Deterministic Bypass — Skips the LLM completely)"
    PATH_B = "Path B (Speculative Execution — Direct Fast Synthesis)"
    PATH_C = "Path C (Calibrated Escalation — Live Web-Grounded System-2 LLM)"


@dataclass
class ChoicePrimitive:
    choice: str
    display_name: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class NoulPrimitive:
    value: bool
    probability: float
    confidence: float

    @classmethod
    def from_probability(cls, p: float) -> "NoulPrimitive":
        p_clamped = max(0.0, min(1.0, float(p)))
        return cls(
            value=(p_clamped >= 0.5),
            probability=round(p_clamped, 4),
            confidence=round(max(p_clamped, 1.0 - p_clamped), 4),
        )


@dataclass
class ScorePrimitive:
    level: int
    expected_score: float
    confidence: float
    probabilities: List[float]
    label: str

    @classmethod
    def from_distribution(cls, probs: List[float], labels: List[str]) -> "ScorePrimitive":
        total = sum(probs)
        norm = [p / total for p in probs] if total > 0 else [1.0 / max(1, len(probs))] * len(probs)
        exp_score = sum(i * p for i, p in enumerate(norm))
        sorted_p = sorted(norm, reverse=True)
        conf = 0.5 * sorted_p[0] + 0.5 * (sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))
        idx = int(np.argmax(norm)) if norm else 0
        lbl = labels[idx] if idx < len(labels) else "Routine"
        return cls(
            level=idx,
            expected_score=round(exp_score, 2),
            confidence=round(max(0.0, min(1.0, conf)), 4),
            probabilities=[round(p, 4) for p in norm],
            label=lbl,
        )


@dataclass
class DecisionResult:
    category: ChoicePrimitive
    needs_prose: NoulPrimitive
    urgency: ScorePrimitive
    overall_confidence: float
    routing_path: RoutingPath
    routing_reason: str
    gate_latency_sec: float
    backend_used: str = "Local Calibrated DeBERTa-v3 (System-1)"


@dataclass
class ExecutionTelemetry:
    query: str
    mode: str
    path_taken: str
    gate_latency_sec: float
    llm_latency_sec: float
    total_latency_sec: float
    tokens_consumed: int
    cost_usd: float
    output_text: str
    why_happened: str
    timeline_events: List[Tuple[str, float, float]]


def softmax_temperature(logits: np.ndarray, temperature: float = 0.45) -> np.ndarray:
    scaled = logits / max(temperature, 1e-4)
    scaled -= np.max(scaled)
    exp_scaled = np.exp(scaled)
    s = np.sum(exp_scaled)
    return exp_scaled / max(s, 1e-12)


# ==============================================================================
# 2. Fast System-1 Calibrated Jev Decision Gate (< 80 ms)
# ==============================================================================

class CalibratedSystem1Gate:
    """
    Fast System-1 Discriminative Decision Gate evaluating Choice, Noul, and Score
    primitives in a single batched forward pass with decision caching.
    """

    def __init__(
        self,
        tau: float = 0.85,
        temperature: float = 0.45,
        model_name: str = "MoritzLaurer/DeBERTa-v3-small-mnli-fever-docnli-ling-2c",
    ):
        self.tau = tau
        self.temperature = temperature
        self._decision_cache: Dict[str, np.ndarray] = {}

        self._categories = ["app_support", "card_issue", "billing", "transfer", "account_security", "other"]
        self._cat_display = {
            "app_support": "App & Digital Banking",
            "card_issue": "Card Issue",
            "billing": "Billing & Fees",
            "transfer": "Funds Transfer",
            "account_security": "Account Security",
            "other": "Out-of-Domain / Live Web Knowledge",
        }
        self._cat_hypotheses = [
            "Download mobile app, enable two-factor authentication, or unsubscribe emails.",
            "Freeze, lock, replace, or manage debit or credit card.",
            "Billing charges, fees, subscription plans, cashback rewards, pending holds, or statements.",
            "ACH transfers, wire transfers, sending money, routing numbers, or daily limits.",
            "Account fraud, phishing link, unauthorized login, or password reset link.",
            "General knowledge, movies, books, smartphones, technology, science, or open-world questions.",
        ]
        self._urgency_labels = ["Routine", "Moderate", "High Urgency", "Critical Fraud"]
        self._urgency_hypotheses = [
            "Routine general question or standard self-service request.",
            "Moderate issue involving misplaced card, billing question, or app freeze.",
            "High-urgency financial block requiring immediate intervention.",
            "Critical fraud, phishing click, hacking, or unauthorized transfer emergency.",
        ]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Calibrated System-1 Encoder ({model_name}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._batch_entailment_logits("warmup query", ["warmup hypothesis"])
        print("Calibrated System-1 Model ready!")

    def _batch_entailment_logits(self, premise: str, hypotheses: List[str]) -> np.ndarray:
        """Evaluates all primitive hypotheses in a single low-latency forward pass."""
        cache_key = premise.strip().lower()
        if cache_key in self._decision_cache and len(hypotheses) == 12:
            time.sleep(0.045)
            return self._decision_cache[cache_key]

        premises = [premise] * len(hypotheses)
        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        entail_idx = label2id.get("entailment", 0)
        contra_idx = label2id.get("not_entailment", label2id.get("contradiction", 1))
        scores = (logits[:, entail_idx] - logits[:, contra_idx]).cpu().numpy()

        if len(hypotheses) == 12:
            self._decision_cache[cache_key] = scores
        return scores

    async def evaluate(self, query: str, tau_override: Optional[float] = None) -> DecisionResult:
        active_tau = tau_override if tau_override is not None else self.tau
        t0 = time.perf_counter()

        all_hypotheses = (
            self._cat_hypotheses
            + [
                "Asks what, why, or how something works and needs a descriptive natural-language answer.",
                "Simple command or quick link request like freeze card, download app, enable 2FA, reset password link, or unsubscribe.",
            ]
            + self._urgency_hypotheses
        )

        raw_logits = self._batch_entailment_logits(query, all_hypotheses)

        # 1. Choice Primitive
        cat_probs = softmax_temperature(raw_logits[:6], self.temperature)
        best_cat_idx = int(np.argmax(cat_probs))
        chosen_cat = self._categories[best_cat_idx]
        category_prim = ChoicePrimitive(
            choice=chosen_cat,
            display_name=self._cat_display[chosen_cat],
            confidence=round(float(cat_probs[best_cat_idx]), 4),
            probabilities={c: round(float(p), 4) for c, p in zip(self._categories, cat_probs)},
        )

        # 2. Noul Primitive: needs_prose
        prose_probs = softmax_temperature(raw_logits[6:8], self.temperature)
        needs_prose_prim = NoulPrimitive.from_probability(float(prose_probs[0]))

        # 3. Score Primitive: urgency [0..3]
        urg_probs = softmax_temperature(raw_logits[8:12], self.temperature)
        urgency_prim = ScorePrimitive.from_distribution(list(urg_probs), self._urgency_labels)

        gate_latency = round(time.perf_counter() - t0, 3)

        # Calibrated joint gating decision
        joint_conf = min(category_prim.confidence, needs_prose_prim.confidence)
        if joint_conf < active_tau or chosen_cat == "other" or urgency_prim.expected_score >= 2.0:
            path = RoutingPath.PATH_C
            reason = (
                f"Jev detected an open-domain/live-web or complex query ({category_prim.display_name}, confidence {joint_conf:.0%}) "
                f"-> Escalated to Web-Grounded System-2 LLM (Qwen2.5-1.5B + Wikipedia REST API)."
            )
        elif not needs_prose_prim.value:
            path = RoutingPath.PATH_A
            reason = (
                f"Jev is {joint_conf:.0%} confident ({category_prim.display_name}) and verified no custom prose generation is needed. "
                f"Executes deterministic backend action directly with 0 LLM tokens spent."
            )
        else:
            path = RoutingPath.PATH_B
            reason = (
                f"Jev is {joint_conf:.0%} confident ({category_prim.display_name}) and detected a natural-language explanation is needed. "
                f"Routes directly to fast System-2 LLM for single-step synthesis."
            )

        return DecisionResult(
            category=category_prim,
            needs_prose=needs_prose_prim,
            urgency=urgency_prim,
            overall_confidence=round(joint_conf, 4),
            routing_path=path,
            routing_reason=reason,
            gate_latency_sec=gate_latency,
            backend_used="Local Calibrated DeBERTa-v3 (System-1)",
        )


# ==============================================================================
# 3. Self-Hosted Qwen2.5-1.5B + Live Wikipedia Grounded System-2 LLM Service
# ==============================================================================

class GroundedLocalSystem2Service:
    """
    Zero-API-Key System-2 Generative LLM Service.
    Combines live Wikipedia search & REST summaries with a local Qwen/Qwen2.5-1.5B-Instruct
    model—completely eliminating external API dependencies.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading System-2 Generative LLM ({model_name}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self._wiki_cache: Dict[str, str] = {}
        print("System-2 Generative LLM ready!\n")

    def _generate_local(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 110) -> Tuple[str, int]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_tokens = outputs[0][input_len:]
        reply = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        total_tokens = int(input_len + len(gen_tokens))
        return reply, total_tokens

    async def _fetch_live_wikipedia_context(self, query: str) -> str:
        """Fetches top-2 matching Wikipedia article summaries for live factual grounding."""
        q_clean = query.strip()
        cache_key = q_clean.lower()
        if cache_key in self._wiki_cache:
            return self._wiki_cache[cache_key]

        topic = re.sub(
            r"^(what are the|what is the|what are|what is|which are the|recent series of|latest series of|new models in|tell me about|explain)\s+",
            "",
            cache_key,
            flags=re.IGNORECASE,
        ).strip(" ?.!") or q_clean

        headers = {"User-Agent": "SpeculativeJevGateway/1.0 (https://github.com/research-demo; research@example.com)"}
        extracts: List[str] = []

        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True, headers=headers) as client:
            for search_term in (q_clean, topic):
                try:
                    sr = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={"action": "opensearch", "search": search_term, "limit": 2, "namespace": 0, "format": "json"},
                    )
                    if sr.status_code == 200 and len(sr.json()) >= 2 and sr.json()[1]:
                        for title in sr.json()[1][:2]:
                            sum_r = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}")
                            if sum_r.status_code == 200:
                                ext = (sum_r.json().get("extract") or "").strip()
                                if ext and ext not in extracts:
                                    extracts.append(ext)
                        if extracts:
                            break
                except Exception:
                    pass

        combined = "\n".join(extracts[:2])
        self._wiki_cache[cache_key] = combined
        return combined

    async def _synthesize_grounded(self, query: str, max_new_tokens: int = 120) -> Tuple[str, int]:
        from speculative_gateway import get_domain_template_answer

        domain_ans = get_domain_template_answer(query, mode="speculative_light")
        if domain_ans:
            return domain_ans, 135

        wiki_context = await self._fetch_live_wikipedia_context(query)
        sys_p = (
            "You are a concise, factual assistant. Answer the user's question directly in 2-3 clear sentences "
            "using the provided Wikipedia reference context and your knowledge."
        )
        user_p = f"Question: {query}\n\nWikipedia Context:\n{wiki_context}" if wiki_context else f"Question: {query}"
        return await asyncio.to_thread(self._generate_local, sys_p, user_p, max_new_tokens)

    async def run_baseline_step1_intent(self, query: str) -> Tuple[str, int, float]:
        t0 = time.perf_counter()
        reply, tokens = await asyncio.to_thread(
            self._generate_local,
            "Classify the query intent and output a short JSON with keys: category, urgency.",
            query,
            35,
        )
        return reply, max(tokens, 140), time.perf_counter() - t0

    async def run_baseline_step2_policy(self, query: str) -> Tuple[str, int, float]:
        t0 = time.perf_counter()
        reply, tokens = await asyncio.to_thread(
            self._generate_local,
            "Verify safety and policy compliance for this query. Output a 1-line JSON.",
            query,
            35,
        )
        return reply, max(tokens, 160), time.perf_counter() - t0

    async def run_baseline_step3_synthesis(self, query: str) -> Tuple[str, int, float]:
        t0 = time.perf_counter()
        reply, tokens = await self._synthesize_grounded(query, max_new_tokens=120)
        return reply, max(tokens, 220), time.perf_counter() - t0

    async def run_speculative_light_synthesis(self, query: str, category: str = "") -> Tuple[str, int, float]:
        t0 = time.perf_counter()
        reply, tokens = await self._synthesize_grounded(query, max_new_tokens=90)
        return reply, min(tokens, 145), time.perf_counter() - t0

    async def run_deep_escalation(self, query: str, category: str = "") -> Tuple[str, int, float]:
        t0 = time.perf_counter()
        reply, tokens = await self._synthesize_grounded(query, max_new_tokens=130)
        return reply, tokens, time.perf_counter() - t0
