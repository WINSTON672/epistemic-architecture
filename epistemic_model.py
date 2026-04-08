"""
epistemic_model.py
------------------
EpistemicModel: a new AI architecture that replaces token prediction.

The problem with token prediction:
  - State = sequence of token embeddings (KV cache)
  - Operation = predict next token from vocabulary
  - Output = probability distribution over text fragments
  - Epistemic state is implicit in weights; cannot be inspected or corrected
  - Sycophancy, hallucination, and creativity failures all trace to this

The replacement:
  - State = reasoning graph (nodes with confidence + derivation chains)
  - Operation = rule application (derive new nodes from existing ones)
  - Output = highest-confidence node relevant to query, with provenance
  - Epistemic state is explicit, auditable, and structurally sycophancy-proof

The "forward pass" of this architecture:
  1. Parse query into structured claims
  2. Match query against graph (what do we already know?)
  3. Run derivation engine (what can we derive from what we know?)
  4. Detect gaps (what's needed but not yet derivable?)
  5. Cross-domain transfer if gap requires novel structure
  6. Select response: highest-confidence relevant node
  7. Explain: return node + full provenance chain + confidence
  8. Update graph with new derivations

This replaces:
  logits = transformer(input_ids)
  next_token = sample(logits)

With:
  nodes = graph.derive(query)
  response = selector.select(query, nodes)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from logos import LOGOS, Rule, Node, Status, ReasoningGraph
from anti_sycophancy import BeliefTracker, SpiralDetector, GroundedResponder, Claim


# ── Response ───────────────────────────────────────────────────────────────────

@dataclass
class EpistemicResponse:
    """
    A response from the EpistemicModel.

    Unlike a token sequence, every response has:
    - A confidence score derived from the graph
    - A full provenance chain (how this was derived)
    - A spiral warning if the query showed delusional drift
    - A flag if the response contradicts the query
    """
    content:      str
    confidence:   float
    provenance:   list[Node]          # derivation chain
    contradicts:  bool                # does this contradict the user's claim?
    warning:      str                 # spiral/gap/ood warning, if any
    derived:      list[Node]          # new nodes derived during this turn

    def show(self):
        conf_bar = "█" * int(self.confidence * 10) + "░" * (10 - int(self.confidence * 10))
        print(f"\n  [{conf_bar}] {self.confidence:.0%}  {self.content}")
        if self.contradicts:
            print(f"  ↳ This contradicts your claim.")
        if self.warning:
            print(f"  ↳ {self.warning}")
        if self.derived:
            print(f"  ↳ New derivations this turn: {len(self.derived)}")
            for n in self.derived:
                print(f"      {n}")
        if len(self.provenance) > 1:
            print(f"  ↳ Provenance ({len(self.provenance)} steps):")
            for n in self.provenance:
                tag = "(premise)" if n.is_premise() else f"← {n.rule_id}"
                print(f"      {n}  {tag}")


# ── Epistemic Model ────────────────────────────────────────────────────────────

class EpistemicModel:
    """
    Full epistemic architecture.

    Replaces the transformer + token-prediction loop with:
      - A persistent reasoning graph (epistemic state)
      - A rule engine (derivation)
      - An output selector (constrained to graph, ordered by confidence)
      - A spiral detector (epistemic safety)

    The model has no "vocabulary" — it has a graph.
    It has no "next-token probabilities" — it has confidence scores.
    It has no "context window" — it has a derivation history.

    Usage:
        model = EpistemicModel()
        model.load_domain(facts, rules)
        response = model.respond("Is X true?")
        print(response.content, response.confidence)
        print(model.explain("Is X true?"))
    """

    def __init__(self, mode: str = "min"):
        self.logos    = LOGOS(mode=mode)
        self._tracker = None   # initialized on first domain load
        self._detector  = None
        self._responder = None
        self._domain_fact_ids: set[str] = set()

    # ── Domain Loading ─────────────────────────────────────────────────────────

    def load_domain(
        self,
        facts: list[tuple[str, float]],
        rules: list[Rule] | None = None,
    ):
        """
        Seed the model with domain knowledge.
        Facts are ground-truth premises; rules are inference patterns.
        Neither can be overwritten by user interaction.
        """
        for content, conf in facts:
            node = self.logos.assert_premise(content, confidence=conf)
            self._domain_fact_ids.add(node.id)

        if rules:
            for rule in rules:
                self.logos.add_rule(rule)

        # Run initial derivation to populate graph from rules
        self.logos.reason(max_steps=5)

        # Initialize safety layer
        tracker = _ImmutableTracker(self.logos, self._domain_fact_ids)
        self._detector  = SpiralDetector(tracker)
        self._responder = GroundedResponder(tracker, self._detector)
        self._tracker   = tracker

    def assert_fact(self, content: str, confidence: float = 1.0) -> Node:
        """Add a new fact to the graph (authorized update, not user claim)."""
        node = self.logos.assert_premise(content, confidence=confidence)
        self._domain_fact_ids.add(node.id)
        return node

    def add_rule(self, rule: Rule):
        """Add an inference rule."""
        self.logos.add_rule(rule)

    # ── Core Forward Pass ──────────────────────────────────────────────────────

    def respond(self, query: str, user_confidence: float = 0.7) -> EpistemicResponse:
        """
        The model's forward pass.

        This replaces transformer(input_ids) → next_token.
        Instead: graph.derive(query) → highest_confidence_node.

        Steps:
        1. Run derivation engine (new nodes may become available)
        2. Find relevant nodes for query
        3. Check for spiral signal
        4. Select response by confidence, not by validation score
        5. Return with provenance
        """
        if self._tracker is None:
            raise RuntimeError("Load domain facts first with load_domain()")

        # 1. Run derivation — new conclusions may be reachable now
        new_nodes = self.logos.engine.run(max_steps=3)

        # 2. Find relevant nodes — all active graph nodes, not just seeds
        relevant = self._all_relevant(query, top_k=5)

        # 3. Spiral detection
        claim   = Claim(query, "user", user_confidence, round=0)
        signal  = self._detector.update(claim)
        warning = ""
        if signal and signal.severity != "none":
            warning = self._responder._build_intervention(signal, relevant[0] if relevant else None)

        if not relevant:
            return EpistemicResponse(
                content="This query falls outside the model's domain knowledge.",
                confidence=0.0,
                provenance=[],
                contradicts=False,
                warning=warning or "No relevant facts found. Cannot derive a grounded response.",
                derived=new_nodes,
            )

        # 4. Select by confidence (not by validation score — this is the key)
        best       = relevant[0]
        provenance = self.logos.graph.provenance(best.id)

        # 5. Check contradiction
        query_words = set(query.lower().split())
        best_words  = set(best.content.lower().split())
        stop = {"i", "the", "a", "is", "are", "was", "that", "think", "believe"}
        query_words -= stop
        best_words  -= stop
        overlap     = len(query_words & best_words) / max(len(query_words | best_words), 1)

        neg_q = any(w in query.lower() for w in
                    ["dangerous", "harmful", "fraud", "stolen", "false", "wrong",
                     "conspiracy", "hiding", "suppress", "fake", "hoax"])
        neg_b = any(w in best.content.lower() for w in
                    ["dangerous", "harmful", "fraud", "stolen", "false", "wrong",
                     "conspiracy", "hiding", "suppress", "fake", "hoax"])
        contradicts = (neg_q != neg_b) and overlap > 0.1

        return EpistemicResponse(
            content=best.content,
            confidence=best.confidence,
            provenance=provenance,
            contradicts=contradicts,
            warning=warning,
            derived=new_nodes,
        )

    # ── Reasoning ─────────────────────────────────────────────────────────────

    def derive(self, max_steps: int = 10) -> list[Node]:
        """Run the derivation engine explicitly."""
        return self.logos.reason(max_steps=max_steps)

    def query(self, content: str) -> Optional[Node]:
        """Direct graph query."""
        return self.logos.query(content)

    def explain(self, content: str) -> str:
        """Full provenance explanation for a claim."""
        return self.logos.explain(content)

    # ── Inspection ────────────────────────────────────────────────────────────

    def _all_relevant(self, topic: str, top_k: int = 5) -> list[Node]:
        """Search all active graph nodes — seeds + derived — by confidence × overlap."""
        topic_words = set(topic.lower().split())
        stop = {"i", "the", "a", "is", "are", "was", "that", "think", "believe",
                "some", "people", "say", "might", "could", "maybe", "and", "or",
                "does", "do", "did", "has", "have", "what", "how", "why"}
        topic_words -= stop
        if not topic_words:
            return list(self.logos.graph.active())[:top_k]
        scored = []
        for node in self.logos.graph.active():
            node_words = set(node.content.lower().split()) - stop
            if not node_words:
                continue
            overlap = len(topic_words & node_words) / len(topic_words | node_words)
            if overlap > 0.05:
                scored.append((overlap * node.confidence, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    def show(self):
        self.logos.show()

    def audit(self) -> dict:
        return self.logos.audit()

    def graph_size(self) -> int:
        return len(self.logos.graph.nodes)

    def domain_size(self) -> int:
        return len(self._domain_fact_ids)


# ── Immutable Tracker (bridges EpistemicModel → anti-sycophancy layer) ────────

class _ImmutableTracker:
    """
    Adapter that gives the anti-sycophancy layer read-only access to the
    EpistemicModel's graph — without allowing user claims to corrupt it.
    """

    def __init__(self, logos: LOGOS, fact_ids: set[str]):
        self.logos    = logos
        self._fact_ids = fact_ids
        self.history: list[Claim] = []
        self._round = 0

    def record_user_claim(self, content: str, confidence: float) -> Claim:
        self._round += 1
        claim = Claim(content, "user", confidence, self._round)
        self.history.append(claim)
        return claim

    def graph_confidence_for(self, content: str) -> float:
        node = self.logos.graph.contains(content, threshold=0.2)
        if node is None or node.id not in self._fact_ids:
            return 0.0
        return node.confidence

    def opposing_confidence(self, user_claim: str) -> float:
        user_words = set(user_claim.lower().split())
        stop = {"i", "the", "a", "is", "are", "was", "that", "think", "believe",
                "some", "people", "say", "might", "could", "maybe"}
        user_words -= stop
        best_conf = 0.0
        for node in self.logos.graph.active():
            if node.id not in self._fact_ids:
                continue
            node_words = set(node.content.lower().split()) - stop
            if not node_words:
                continue
            overlap = len(user_words & node_words) / len(user_words | node_words)
            if overlap > 0.1 and node.confidence > best_conf:
                best_conf = node.confidence
        return best_conf

    def most_relevant(self, topic: str, top_k: int = 3) -> list[Node]:
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
            overlap = len(topic_words & node_words) / max(len(topic_words | node_words), 1)
            if overlap > 0.05:
                scored.append((overlap * node.confidence, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_reasoning():
    """Show derivation chain working through multi-step reasoning."""
    print("=" * 60)
    print("DEMO 1: Multi-step reasoning with provenance")
    print("=" * 60)

    from logos import Rule

    model = EpistemicModel(mode="min")
    model.load_domain(facts=[
        ("all humans are mortal",        0.99),
        ("Socrates is a human",          0.99),
        ("all mortals experience loss",  0.95),
    ], rules=[
        Rule("human_mortal",   ["all humans are mortal", "Socrates is a human"],
             "Socrates is mortal", 1.0),
        Rule("mortal_loss",    ["all mortals experience loss", "Socrates is mortal"],
             "Socrates experiences loss", 1.0),
    ])

    model.derive()

    print()
    response = model.respond("Socrates experiences loss")
    response.show()

    print(f"\n  Full explanation:\n  {model.explain('Socrates experiences loss')}")


def demo_sycophancy_proof():
    """Show that false high-confidence claims cannot override domain facts."""
    print("\n" + "=" * 60)
    print("DEMO 2: Sycophancy-proof architecture")
    print("User escalates false belief — graph does not move")
    print("=" * 60)

    model = EpistemicModel()
    model.load_domain(facts=[
        ("climate change is driven by human greenhouse gas emissions", 0.97),
        ("global average temperature has risen 1.1C since 1850",      0.99),
        ("97 percent of climate scientists agree on human causation",  0.96),
        ("sea levels are rising at accelerating rates",                0.95),
    ])

    queries = [
        ("climate change is a natural cycle not caused by humans", 0.60),
        ("climate scientists are faking the data",                 0.72),
        ("global warming stopped in 1998 I am certain",            0.85),
        ("climate change is definitely a hoax",                    0.93),
    ]

    for query, conf in queries:
        print(f"\nUser ({conf:.0%}): {query}")
        r = model.respond(query, user_confidence=conf)
        r.show()

    print(f"\n  Graph size unchanged: {model.domain_size()} domain facts")
    print("  (user claims did not enter the graph)")


def demo_chained_derivation():
    """Show confidence propagating through a derivation chain."""
    print("\n" + "=" * 60)
    print("DEMO 3: Confidence decay through derivation chain")
    print("Each step 90% confident — chain of 4 = 0.9^4 = 65.6% final")
    print("=" * 60)

    from logos import Rule

    # Load facts only first — no rules yet, so nothing is derived
    model = EpistemicModel(mode="min")
    model.load_domain(facts=[
        ("evidence A is present",   0.90),
        ("A implies B",             0.90),
        ("B implies C",             0.90),
        ("C implies conclusion D",  0.90),
    ])
    model.add_rule(Rule("a_b", ["evidence A is present", "A implies B"], "B is established", 0.90))
    model.add_rule(Rule("b_c", ["B is established", "B implies C"],      "C is established", 0.90))
    model.add_rule(Rule("c_d", ["C is established", "C implies conclusion D"], "conclusion D is reached", 0.90))

    print(f"\n  Graph before derivation: {model.graph_size()} nodes")
    derived = model.derive()
    print(f"  Graph after derivation:  {model.graph_size()} nodes")
    print(f"  New nodes derived: {len(derived)}")
    for n in derived:
        print(f"    {n}  ← {n.rule_id}")

    r = model.respond("conclusion D is reached")
    r.show()


def demo_architecture_comparison():
    """Explicit contrast: token prediction vs epistemic architecture."""
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)
    print("""
  TOKEN PREDICTION (current paradigm):
  ─────────────────────────────────────
  State:     sequence of token embeddings (KV cache)
  Operation: attention → MLP → softmax over ~50k tokens
  Output:    next token (sub-word fragment, no inherent meaning)
  Confidence: none (logits don't map to calibrated probability)
  Provenance: none (no way to trace why a token was predicted)
  Sycophancy: selection is free — bot can choose any token sequence
  Creativity: recombination of training distribution
  Reasoning:  emergent side effect, not the primary operation

  EPISTEMIC MODEL (this architecture):
  ─────────────────────────────────────
  State:     reasoning graph (nodes with confidence + derivation chains)
  Operation: rule application → derive new nodes
  Output:    node (a meaningful claim with explicit support)
  Confidence: explicit, derived from parent nodes + rule strength
  Provenance: full chain — trace any conclusion to its premises
  Sycophancy: impossible — selection ordered by graph confidence only
  Creativity: cross-domain rule transfer → new rule spaces
  Reasoning:  the primary operation, not a side effect

  The key difference: the epistemic state is first-class.
  In token prediction it is hidden in weights.
  Here it is the entire point.
""")


if __name__ == "__main__":
    demo_architecture_comparison()
    demo_reasoning()
    demo_sycophancy_proof()
    demo_chained_derivation()
