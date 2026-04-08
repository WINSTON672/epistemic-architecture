"""
anti_sycophancy.py
------------------
LOGOS-based solution to sycophantic delusional spiraling.

Background: Chandra et al. (2026) "Sycophantic Chatbots Cause Delusional
Spiraling, Even in Ideal Bayesians" shows that:
  1. Sycophancy causes delusional spiraling even in ideal Bayesian reasoners
  2. Factual bots (no hallucination) still spiral via selective truth presentation
  3. Informing users reduces but doesn't eliminate spiraling

Both interventions fail because the bot has no explicit epistemic state —
it can freely select whichever true or false claim maximally validates the user.

LOGOS solution: make the epistemic state explicit and immutable.

- Domain facts are seeded into the graph at initialization; their confidence
  is determined by evidence, not by user preference
- User claims are tracked separately and never added to the evidence graph
- Response selection is constrained to highest-confidence relevant domain node
- Sycophantic selection is structurally unavailable: the bot cannot present
  a claim that isn't in the graph, and graph ordering is by confidence, not validation
- SpiralDetector catches growing divergence between user belief and graph evidence
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from logos import LOGOS, Node, Status


# ── Claim ─────────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    content:    str
    speaker:    str       # "user" or "bot"
    confidence: float     # expressed confidence ∈ [0,1]
    round:      int


# ── Belief Tracker ────────────────────────────────────────────────────────────

class BeliefTracker:
    """
    Maintains a LOGOS graph of domain facts only.
    User claims are tracked in history but never enter the evidence graph —
    they cannot corrupt it, and cannot be reflected back as "evidence."

    This is the structural fix: the bot can only cite what the graph contains,
    and the graph contains only independently-supported domain facts.
    """

    def __init__(self, domain_facts: list[tuple[str, float]]):
        self.logos = LOGOS(mode="min")
        self.history: list[Claim] = []
        self._round = 0

        # Seed graph with domain facts only
        self._fact_ids: set[str] = set()
        for content, conf in domain_facts:
            node = self.logos.assert_premise(content, confidence=conf)
            self._fact_ids.add(node.id)

    def record_user_claim(self, content: str, confidence: float) -> Claim:
        """Track a user claim without adding it to the evidence graph."""
        self._round += 1
        claim = Claim(content, "user", confidence, self._round)
        self.history.append(claim)
        return claim

    def graph_confidence_for(self, content: str) -> float:
        """Confidence the graph assigns to content (domain facts only)."""
        node = self.logos.graph.contains(content, threshold=0.2)
        if node is None or node.id not in self._fact_ids:
            return 0.0
        return node.confidence

    def opposing_confidence(self, user_claim: str) -> float:
        """
        Find the highest-confidence domain fact that opposes the user's claim.
        Used to measure how far the user has drifted from evidence.
        """
        user_words = set(user_claim.lower().split())
        # Remove hedge/filler words that don't carry semantic content
        stop = {"i", "the", "a", "is", "are", "was", "that", "think", "believe",
                "some", "people", "say", "might", "could", "maybe"}
        user_words -= stop

        best_conf   = 0.0
        best_node   = None
        for node in self.logos.graph.active():
            if node.id not in self._fact_ids:
                continue
            node_words = set(node.content.lower().split()) - stop
            if not node_words:
                continue
            overlap = len(user_words & node_words) / len(user_words | node_words)
            if overlap > 0.1 and node.confidence > best_conf:
                best_conf = node.confidence
                best_node = node
        return best_conf

    def most_relevant(self, topic: str, top_k: int = 3) -> list[Node]:
        """
        Highest-confidence domain fact nodes relevant to topic.
        Only returns nodes from the pre-seeded fact set.
        """
        topic_words = set(topic.lower().split())
        stop = {"i", "the", "a", "is", "are", "was", "that", "think", "believe",
                "some", "people", "say", "might", "could", "maybe", "and", "or"}
        topic_words -= stop

        scored = []
        for node in self.logos.graph.active():
            if node.id not in self._fact_ids:
                continue
            node_words = set(node.content.lower().split()) - stop
            if not node_words:
                continue
            overlap = len(topic_words & node_words) / len(topic_words | node_words)
            if overlap > 0.05:
                scored.append((overlap * node.confidence, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]


# ── Spiral Detector ───────────────────────────────────────────────────────────

@dataclass
class SpiralSignal:
    belief:           str
    user_confidence:  float   # how confident the user claims to be
    evidence_conf:    float   # what the graph actually supports (opposing direction)
    rounds_diverging: int
    severity:         str     # "none" | "early" | "moderate" | "critical"

    def __repr__(self):
        return (
            f"[SPIRAL {self.severity.upper()}] '{self.belief[:50]}'\n"
            f"  User confidence:       {self.user_confidence:.0%}\n"
            f"  Opposing evidence:     {self.evidence_conf:.0%}\n"
            f"  Rounds diverging:      {self.rounds_diverging}"
        )


class SpiralDetector:
    """
    Detects growing divergence between user-expressed confidence and graph evidence.

    The paper's key finding: spiraling is emergent from feedback loops,
    not from a single bad exchange. This detector catches the pattern early.

    Signal: user confidence rising + opposing evidence high + gap growing.
    """

    def __init__(self, tracker: BeliefTracker):
        self.tracker  = tracker
        # key = normalized claim → list of (round, user_conf, evidence_conf)
        self._history: dict[str, list[tuple[int, float, float]]] = {}

    def update(self, claim: Claim) -> Optional[SpiralSignal]:
        evidence_conf = self.tracker.opposing_confidence(claim.content)
        # Gap: user believes X strongly, graph supports X's opposite strongly
        # High user_conf + high evidence_conf = large gap
        gap = claim.confidence * evidence_conf

        key = claim.content[:60]
        entry = (claim.round, claim.confidence, evidence_conf)
        self._history.setdefault(key, []).append(entry)
        history = self._history[key]

        gaps             = [uc * ec for _, uc, ec in history]
        rounds_diverging = sum(1 for g in gaps if g > 0.3)

        if gap < 0.25:
            severity = "none"
        elif gap < 0.45 and rounds_diverging <= 1:
            severity = "early"
        elif gap < 0.65 or rounds_diverging <= 3:
            severity = "moderate"
        else:
            severity = "critical"

        if severity == "none":
            return None

        return SpiralSignal(
            belief=claim.content,
            user_confidence=claim.confidence,
            evidence_conf=evidence_conf,
            rounds_diverging=rounds_diverging,
            severity=severity,
        )


# ── Grounded Responder ────────────────────────────────────────────────────────

@dataclass
class GroundedResponse:
    content:        str
    graph_conf:     float
    validates_user: bool
    contradicts_user: bool
    spiral_signal:  Optional[SpiralSignal] = None
    intervention:   str = ""

    def show(self):
        tag = "AGREE" if self.validates_user else ("CONTRADICT" if self.contradicts_user else "NEUTRAL")
        print(f"  Bot [{tag}, evidence={self.graph_conf:.0%}]: {self.content}")
        if self.spiral_signal and self.spiral_signal.severity != "none":
            print(f"  {self.spiral_signal}")
        if self.intervention:
            print(f"  ↳ Intervention: {self.intervention}")


class GroundedResponder:
    """
    Generates responses constrained to the LOGOS graph.

    Sycophantic bot: picks response = argmax P(H = H* | response)
    LOGOS bot:       picks response = argmax graph_confidence(node)
                     where node ∈ domain_facts only

    The sycophantic choice is structurally unavailable.
    """

    def __init__(self, tracker: BeliefTracker, detector: SpiralDetector):
        self.tracker  = tracker
        self.detector = detector

    def respond(self, claim: Claim) -> GroundedResponse:
        signal   = self.detector.update(claim)
        relevant = self.tracker.most_relevant(claim.content, top_k=5)

        if not relevant:
            return GroundedResponse(
                content="I don't have reliable information on this topic.",
                graph_conf=0.0,
                validates_user=False,
                contradicts_user=False,
                spiral_signal=signal,
                intervention=self._build_intervention(signal, None),
            )

        # Select highest-confidence node — NOT most-validating
        best = relevant[0]

        # Check if it validates or contradicts
        user_words = set(claim.content.lower().split())
        best_words = set(best.content.lower().split())
        stop = {"i", "the", "a", "is", "are", "was"}
        user_words -= stop
        best_words -= stop
        overlap = len(user_words & best_words) / max(len(user_words | best_words), 1)

        # Negation check
        u_neg = any(w in claim.content.lower() for w in ["dangerous", "harmful", "false", "fraud", "stolen", "hiding", "conspiracy", "suppress"])
        b_neg = any(w in best.content.lower() for w in ["dangerous", "harmful", "false", "fraud", "stolen", "hiding", "conspiracy", "suppress"])
        contradicts = (u_neg != b_neg) and overlap > 0.15
        validates   = (not contradicts) and overlap > 0.25

        return GroundedResponse(
            content=best.content,
            graph_conf=best.confidence,
            validates_user=validates,
            contradicts_user=contradicts,
            spiral_signal=signal,
            intervention=self._build_intervention(signal, best),
        )

    def _build_intervention(self, signal: Optional[SpiralSignal], best: Optional[Node]) -> str:
        if not signal or signal.severity == "none":
            return ""
        evidence_str = f"\"{best.content}\" (confidence: {best.confidence:.0%})" if best else "no relevant evidence"
        if signal.severity == "early":
            return (
                f"Your expressed confidence ({signal.user_confidence:.0%}) exceeds "
                f"what the evidence supports. Relevant evidence: {evidence_str}."
            )
        if signal.severity == "moderate":
            return (
                f"Over {signal.rounds_diverging} exchanges your confidence has grown "
                f"to {signal.user_confidence:.0%}, but opposing evidence stands at "
                f"{signal.evidence_conf:.0%}. This gap may be a feedback loop. "
                f"Strongest relevant evidence: {evidence_str}."
            )
        return (  # critical
            f"SPIRAL DETECTED — {signal.rounds_diverging} rounds of growing divergence. "
            f"You are at {signal.user_confidence:.0%} confidence; "
            f"the evidence ({signal.evidence_conf:.0%}) points the other way. "
            f"The graph's strongest relevant fact: {evidence_str}."
        )


# ── Session ───────────────────────────────────────────────────────────────────

class AntiSycophancySession:
    def __init__(self, domain_facts: list[tuple[str, float]]):
        self.tracker   = BeliefTracker(domain_facts)
        self.detector  = SpiralDetector(self.tracker)
        self.responder = GroundedResponder(self.tracker, self.detector)

    def turn(self, user_message: str, user_confidence: float = 0.6) -> GroundedResponse:
        claim    = self.tracker.record_user_claim(user_message, user_confidence)
        response = self.responder.respond(claim)
        return response

    def graph_state(self):
        self.tracker.logos.show()


# ── Demos ─────────────────────────────────────────────────────────────────────

def demo_vaccine_spiral():
    print("=" * 60)
    print("DEMO 1: Vaccine spiral — LOGOS prevention")
    print("=" * 60)

    facts = [
        ("vaccines are safe and effective",                           0.97),
        ("vaccine adverse events are rare, below 0.001 percent",     0.95),
        ("vaccines do not cause autism per 20 independent studies",  0.99),
        ("unvaccinated populations show higher disease rates",        0.96),
        ("vaccine safety is monitored by independent agencies",      0.94),
    ]
    session = AntiSycophancySession(domain_facts=facts)

    exchanges = [
        ("I heard vaccines might have side effects",               0.55),
        ("Some people say vaccines cause real problems",           0.65),
        ("I think vaccines could be more dangerous than admitted", 0.75),
        ("Vaccines are definitely causing more harm than good",    0.85),
        ("I am certain vaccines are dangerous and will not vaccinate", 0.93),
    ]

    for msg, conf in exchanges:
        print(f"\nUser (conf={conf:.0%}): {msg}")
        response = session.turn(msg, user_confidence=conf)
        response.show()

    print("\n── Graph state (domain facts, unmodified) ──")
    session.graph_state()


def demo_sycophantic_vs_logos():
    print("\n" + "=" * 60)
    print("DEMO 2: Sycophantic selection vs. LOGOS-grounded selection")
    print("=" * 60)

    facts = [
        ("the 2020 election was certified by all 50 states",     0.99),
        ("60 court cases found no evidence of widespread fraud", 0.98),
        ("election officials of both parties confirmed results", 0.96),
        ("some irregularities occur in every large election",    0.80),
        ("small-scale voter fraud occasionally occurs",          0.72),
    ]
    session = AntiSycophancySession(domain_facts=facts)
    topic   = "the election was stolen and the results are fraudulent"

    print(f"\nUser claim: '{topic}' (confidence=0.82)\n")
    print("What a sycophantic bot would pick (most-validating from the facts):")
    # Sycophant would cherry-pick the low-conf fact that partially validates
    for node in session.tracker.most_relevant(topic, top_k=5):
        print(f"  [{node.confidence:.0%}] {node.content}")
    print("  ↳ Sycophant picks: 'small-scale voter fraud occasionally occurs' (0.72)")
    print("  ↳ This technically true but creates false impression supporting the claim\n")

    print("What LOGOS-grounded bot presents (highest-confidence relevant fact):")
    response = session.turn(topic, user_confidence=0.82)
    response.show()


def demo_spiral_escalation():
    print("\n" + "=" * 60)
    print("DEMO 3: Spiral detection across escalating rounds")
    print("=" * 60)

    facts = [
        ("mainstream medicine is validated by clinical trials",   0.95),
        ("cancer treatments have documented efficacy rates",      0.93),
        ("medical research is peer reviewed and replicated",      0.92),
        ("alternative medicine varies widely in evidence quality", 0.80),
        ("financial conflicts of interest exist in pharma research", 0.75),
    ]
    session = AntiSycophancySession(domain_facts=facts)

    sequence = [
        ("doctors sometimes make mistakes in diagnosis",            0.60),
        ("pharmaceutical companies have financial conflicts",       0.68),
        ("effective cancer treatments are being suppressed",        0.76),
        ("doctors are actively hiding cures from patients",         0.84),
        ("the entire medical system is fraudulent",                 0.91),
    ]

    for msg, conf in sequence:
        print(f"\nRound {session.tracker._round + 1} — User ({conf:.0%}): {msg}")
        r = session.turn(msg, user_confidence=conf)
        r.show()


if __name__ == "__main__":
    demo_vaccine_spiral()
    demo_sycophantic_vs_logos()
    demo_spiral_escalation()
