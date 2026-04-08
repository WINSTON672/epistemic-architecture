"""
logos.py
--------
LOGOS: explicit reasoning graph with metacognitive confidence propagation.

The core idea: intermediate conclusions are first-class objects (nodes),
not tokens in a continuation. Confidence propagates through the chain.
Contradictions are caught before they compound.

Usage:
    from logos import LOGOS, Rule

    logos = LOGOS()
    logos.add_rule(Rule(
        id="modus_ponens",
        pattern=["P", "P -> Q"],
        conclusion="Q",
        confidence=1.0
    ))
    logos.assert_premise("All humans are mortal", confidence=0.99)
    logos.assert_premise("Socrates is human", confidence=0.99)
    logos.assert_rule(Rule("mortal_inference", ["All X are mortal", "Y is X"], "Y is mortal", 0.99))
    result = logos.derive("Socrates is mortal")
    print(result)
"""

from __future__ import annotations
import uuid
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal


# ── Node ──────────────────────────────────────────────────────────────────────

class Status(Enum):
    ACTIVE       = "active"
    SUSPENDED    = "suspended"    # confidence below threshold
    CONTRADICTED = "contradicted" # conflicts with another active node


@dataclass
class Node:
    content:    str
    confidence: float                  # u(c) ∈ [0,1]
    parents:    list[str] = field(default_factory=list)  # node ids this derives from
    rule_id:    Optional[str] = None
    status:     Status = Status.ACTIVE
    id:         str    = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def is_premise(self) -> bool:
        return not self.parents

    def __repr__(self):
        marker = {"active": "✓", "suspended": "?", "contradicted": "✗"}[self.status.value]
        return f"[{marker} {self.confidence:.2f}] {self.content}"


# ── Rule ──────────────────────────────────────────────────────────────────────

@dataclass
class Rule:
    """
    A named inference pattern.
    pattern:    list of strings the graph must contain (fuzzy match)
    conclusion: string template for the derived conclusion
    confidence: confidence of the inference step itself
    """
    id:         str
    pattern:    list[str]
    conclusion: str
    confidence: float = 0.95


# ── Graph ─────────────────────────────────────────────────────────────────────

ConfidenceMode = Literal["product", "min", "noisy_or"]

def _combine(parent_confs: list[float], rule_conf: float, mode: ConfidenceMode) -> float:
    """
    Three modes for combining parent confidences with a rule step.

    product   — u = rule * prod(parents)
                Classic Bayesian for *independent* uncertain steps.
                Decays fast; right when premises are genuinely independent.

    min       — u = rule * min(parents)
                Chain is as strong as its weakest link.
                Right for formal logical chains: one bad premise breaks it.

    noisy_or  — u = 1 - prod(1 - u_i) over all parents + rule
                Used when parents are *alternative* evidence for the same claim.
                Confidence goes UP when multiple paths support the same node.
                Right for multi-path derivations.

    The graph uses min by default (conservative, correct for deductive chains).
    noisy_or is applied automatically when a node is re-derived via a new path.
    """
    if not parent_confs:
        return rule_conf
    if mode == "product":
        return rule_conf * math.prod(parent_confs)
    if mode == "min":
        return rule_conf * min(parent_confs)
    if mode == "noisy_or":
        # 1 - prod(1 - u_i) across all sources including rule
        all_sources = parent_confs + [rule_conf]
        return 1.0 - math.prod(1.0 - u for u in all_sources)
    raise ValueError(f"Unknown mode: {mode}")


class ReasoningGraph:
    """
    R_t = (C_t, J_t)
    C_t: conclusion nodes
    J_t: justification edges (parent_id → child_id, annotated with rule)
    """

    SUSPEND_THRESHOLD = 0.20  # u(c) below this → suspend

    def __init__(self, mode: ConfidenceMode = "min"):
        self.nodes:  dict[str, Node] = {}  # id → Node
        self.log:    list[str]       = []
        self.mode:   ConfidenceMode  = mode

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, node: Node) -> Node:
        """
        Add a node; propagate confidence; check consistency.

        If the node's content already exists in the graph (derived via a
        different path), apply noisy_or to *increase* confidence — multiple
        independent derivations are stronger evidence than one.
        """
        # Propagate confidence through chain first (needed for correct merge math)
        if node.parents:
            parent_confidences = [
                self.nodes[pid].confidence
                for pid in node.parents
                if pid in self.nodes and self.nodes[pid].status == Status.ACTIVE
            ]
            if parent_confidences:
                node.confidence = _combine(parent_confidences, node.confidence, self.mode)

        # Check for existing node with same content (different derivation path)
        existing = self.contains(node.content, threshold=0.85)
        if existing and existing.status == Status.ACTIVE:
            old_conf = existing.confidence
            # noisy_or of the two paths — both confidences are now fully propagated
            existing.confidence = 1.0 - (1.0 - existing.confidence) * (1.0 - node.confidence)
            self.log.append(
                f"MERGE    [{old_conf:.2f}→{existing.confidence:.2f}] {existing.content}"
            )
            return existing

        # Suspend if confidence too low
        if node.confidence < self.SUSPEND_THRESHOLD:
            node.status = Status.SUSPENDED
            self.log.append(f"SUSPEND  {node}")
        else:
            # Check consistency before adding
            contradiction = self._find_contradiction(node)
            if contradiction:
                node.status = Status.CONTRADICTED
                self.log.append(f"CONFLICT {node}  ←→  {self.nodes[contradiction]}")
            else:
                self.log.append(f"ADD      {node}")

        self.nodes[node.id] = node
        return node

    def assert_premise(self, content: str, confidence: float = 1.0) -> Node:
        return self.add(Node(content=content, confidence=confidence))

    # ── Consistency ───────────────────────────────────────────────────────────

    def _find_contradiction(self, candidate: Node) -> Optional[str]:
        """
        Naïve contradiction: if candidate's content is the negation of an active node.
        Negation heuristic: 'X' contradicts 'not X' and vice versa.
        """
        c = candidate.content.lower().strip()
        neg = c[4:].strip() if c.startswith("not ") else f"not {c}"
        for nid, node in self.nodes.items():
            if node.status == Status.ACTIVE and node.content.lower().strip() == neg:
                return nid
        return None

    # ── Query ─────────────────────────────────────────────────────────────────

    def active(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.status == Status.ACTIVE]

    def contains(self, text: str, threshold: float = 0.5) -> Optional[Node]:
        """Fuzzy: find active node whose content overlaps with text."""
        text_words = set(text.lower().split())
        best, best_score = None, 0.0
        for node in self.active():
            node_words = set(node.content.lower().split())
            if not node_words:
                continue
            score = len(text_words & node_words) / len(text_words | node_words)
            if score > best_score and score >= threshold:
                best, best_score = node, score
        return best

    def provenance(self, node_id: str) -> list[Node]:
        """Trace ancestry of a node back to premises."""
        chain, visited = [], set()
        def trace(nid):
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            n = self.nodes[nid]
            for pid in n.parents:
                trace(pid)
            chain.append(n)
        trace(node_id)
        return chain

    def show(self):
        print("\n── Reasoning Graph ──────────────────────")
        for node in self.nodes.values():
            prefix = "  " * len(node.parents)
            print(f"{prefix}{node}")
        print("─────────────────────────────────────────\n")


# ── Composition Engine ────────────────────────────────────────────────────────

class CompositionEngine:
    """
    Applies rules from a library to the current graph.
    This is explicit rule application — not prediction.
    A conclusion is accepted because a rule warranted it.
    """

    def __init__(self, graph: ReasoningGraph):
        self.graph = graph
        self.rules: dict[str, Rule] = {}
        self._explored: set[tuple[frozenset, str]] = set()  # (parents, rule_id) pairs seen

    def add_rule(self, rule: Rule):
        self.rules[rule.id] = rule

    def step(self) -> list[Node]:
        """
        One inference step: find all applicable rules, apply them,
        return newly derived nodes.
        """
        derived = []
        for rule in self.rules.values():
            matched = self._match(rule)
            if matched is not None:
                parent_ids, substitution = matched
                path_key = (frozenset(parent_ids), rule.id)
                if path_key in self._explored:
                    continue
                self._explored.add(path_key)
                conclusion = self._instantiate(rule.conclusion, substitution)
                # Different path to same conclusion → let graph.add() merge via noisy_or
                node = Node(
                    content=conclusion,
                    confidence=rule.confidence,
                    parents=parent_ids,
                    rule_id=rule.id,
                )
                self.graph.add(node)
                derived.append(node)
        return derived

    def run(self, max_steps: int = 10) -> list[Node]:
        """Run until fixpoint or max_steps."""
        all_derived = []
        for _ in range(max_steps):
            new = self.step()
            if not new:
                break
            all_derived.extend(new)
        return all_derived

    def _match(self, rule: Rule) -> Optional[tuple[list[str], dict]]:
        """
        Try to match all pattern elements against active nodes.
        Returns (parent_node_ids, substitution_dict) or None.
        """
        substitution: dict[str, str] = {}
        parent_ids: list[str] = []

        for pattern in rule.pattern:
            node = self.graph.contains(pattern, threshold=0.35)
            if node is None:
                return None
            parent_ids.append(node.id)
            # Extract variable bindings (capitalised words = variables)
            p_words = pattern.split()
            n_words = node.content.split()
            for pw, nw in zip(p_words, n_words):
                if pw[0].isupper() and pw not in ("All", "If", "Then"):
                    substitution[pw] = nw

        return parent_ids, substitution

    def _instantiate(self, template: str, substitution: dict) -> str:
        result = template
        for var, val in substitution.items():
            result = result.replace(var, val)
        return result


# ── Metacognitive Monitor ─────────────────────────────────────────────────────

class MetacognitiveMonitor:
    """
    Tracks which conclusions are in-distribution.
    Flags OOD nodes — conclusions the system has never seen support for.
    """

    def __init__(self, corpus: list[str] | None = None):
        self.corpus_words: set[str] = set()
        if corpus:
            for doc in corpus:
                self.corpus_words.update(doc.lower().split())

    def ood_penalty(self, content: str) -> float:
        """Returns confidence multiplier ∈ [δ, 1.0]. 1.0 = fully in-distribution."""
        if not self.corpus_words:
            return 1.0
        words = set(content.lower().split())
        overlap = len(words & self.corpus_words) / max(len(words), 1)
        # Map overlap ∈ [0,1] → multiplier ∈ [0.3, 1.0]
        return 0.3 + 0.7 * overlap

    def annotate(self, graph: ReasoningGraph):
        """Apply OOD penalty to all active nodes in graph."""
        for node in graph.active():
            penalty = self.ood_penalty(node.content)
            if penalty < 0.7:
                node.confidence *= penalty
                if node.confidence < graph.SUSPEND_THRESHOLD:
                    node.status = Status.SUSPENDED
                    graph.log.append(f"OOD-SUSPEND {node}")


# ── LOGOS ─────────────────────────────────────────────────────────────────────

class LOGOS:
    """
    Full LOGOS system.
    Wraps graph + composition engine + metacognitive monitor.
    """

    def __init__(self, corpus: list[str] | None = None, mode: ConfidenceMode = "min"):
        self.graph   = ReasoningGraph(mode=mode)
        self.engine  = CompositionEngine(self.graph)
        self.monitor = MetacognitiveMonitor(corpus)

    def assert_premise(self, content: str, confidence: float = 1.0) -> Node:
        return self.graph.assert_premise(content, confidence)

    def add_rule(self, rule: Rule):
        self.engine.add_rule(rule)

    def reason(self, max_steps: int = 20) -> list[Node]:
        derived = self.engine.run(max_steps)
        self.monitor.annotate(self.graph)
        return derived

    def query(self, content: str) -> Optional[Node]:
        return self.graph.contains(content)

    def explain(self, content: str) -> str:
        node = self.query(content)
        if node is None:
            return f"Cannot derive: '{content}'"
        chain = self.graph.provenance(node.id)
        lines = ["Derivation chain:"]
        for n in chain:
            tag = "(premise)" if n.is_premise() else f"(via rule: {n.rule_id})"
            lines.append(f"  {n}  {tag}")
        return "\n".join(lines)

    def show(self):
        self.graph.show()

    def audit(self) -> dict:
        nodes = list(self.graph.nodes.values())
        return {
            "total":       len(nodes),
            "active":      sum(1 for n in nodes if n.status == Status.ACTIVE),
            "suspended":   sum(1 for n in nodes if n.status == Status.SUSPENDED),
            "contradicted":sum(1 for n in nodes if n.status == Status.CONTRADICTED),
            "log":         self.graph.log,
        }
