"""
APPLUS3 Anchored Multi-Agent Review Module
Based on Idea2Paper's Anchored Multi-Agent Review methodology

Validates generated code against anchor examples using relative comparisons.
Three reviewers evaluate: CodeQuality, InterfaceConsistency, PatternAdherence
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .knowledge_graph import KnowledgeGraph, Node


class ReviewDimension(str, Enum):
    """Dimensions for code review."""
    CODE_QUALITY = "CodeQuality"
    INTERFACE_CONSISTENCY = "InterfaceConsistency"
    PATTERN_ADHERENCE = "PatternAdherence"


class Judgement(str, Enum):
    """Comparison judgement options."""
    BETTER = "better"
    TIE = "tie"
    WORSE = "worse"


@dataclass
class Comparison:
    """Single comparison against an anchor."""
    anchor_id: str
    anchor_score: float
    judgement: Judgement
    confidence: float
    rationale: str


@dataclass
class ReviewResult:
    """Result from a single reviewer."""
    dimension: ReviewDimension
    score: float
    comparisons: List[Comparison]
    feedback: str
    loss: float = 0.0


@dataclass
class AnchoredReviewResult:
    """Complete review result from all reviewers."""
    passed: bool
    avg_score: float
    reviews: List[ReviewResult]
    main_issue: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anchor:
    """An anchor example for comparison."""
    id: str
    name: str
    score10: float
    weight: float
    code_snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnchoredReviewer:
    """
    Multi-agent reviewer using anchored comparisons.

    Instead of direct scoring, compares generated code against anchor examples
    and uses deterministic fitting to compute final scores.
    """

    # Sigmoid steepness parameter
    SIGMOID_K = 1.2
    # Grid search step for score fitting
    GRID_STEP = 0.05
    # Default pass threshold
    PASS_SCORE = 6.5

    def __init__(self, kg: Optional[KnowledgeGraph] = None):
        """
        Initialize reviewer.

        Args:
            kg: Optional KnowledgeGraph for anchor retrieval
        """
        self.kg = kg
        self.reviewers = [
            {'name': 'Reviewer A', 'dimension': ReviewDimension.CODE_QUALITY,
             'focus': 'code correctness, error handling, edge cases'},
            {'name': 'Reviewer B', 'dimension': ReviewDimension.INTERFACE_CONSISTENCY,
             'focus': 'API compatibility, type consistency, naming conventions'},
            {'name': 'Reviewer C', 'dimension': ReviewDimension.PATTERN_ADHERENCE,
             'focus': 'design patterns, architecture alignment, best practices'},
        ]

    def review(
        self,
        generated_code: str,
        context: Optional[Dict[str, Any]] = None,
        anchors: Optional[List[Anchor]] = None
    ) -> AnchoredReviewResult:
        """
        Review generated code using anchored comparisons.

        Args:
            generated_code: The code to review
            context: Optional context (target node info, requirements)
            anchors: Optional list of anchor examples for comparison

        Returns:
            AnchoredReviewResult with scores and feedback
        """
        context = context or {}

        # Get or create anchors
        if not anchors:
            anchors = self._get_default_anchors(context)

        if not anchors:
            # Fallback to neutral scoring
            return self._neutral_result("No anchors available for comparison")

        # Run each reviewer
        reviews: List[ReviewResult] = []
        for reviewer in self.reviewers:
            review = self._run_reviewer(
                reviewer=reviewer,
                code=generated_code,
                anchors=anchors,
                context=context
            )
            reviews.append(review)

        # Calculate final result
        return self._compute_final_result(reviews, anchors, context)

    def _run_reviewer(
        self,
        reviewer: Dict[str, Any],
        code: str,
        anchors: List[Anchor],
        context: Dict[str, Any]
    ) -> ReviewResult:
        """
        Run a single reviewer's evaluation.

        Args:
            reviewer: Reviewer configuration
            code: Code to review
            anchors: Anchor examples
            context: Generation context

        Returns:
            ReviewResult for this dimension
        """
        dimension = reviewer['dimension']
        comparisons: List[Comparison] = []

        # Compare against each anchor
        for anchor in anchors:
            comparison = self._compare_to_anchor(
                code=code,
                anchor=anchor,
                dimension=dimension,
                focus=reviewer['focus']
            )
            comparisons.append(comparison)

        # Compute score from comparisons
        score, loss = self._compute_score_from_comparisons(anchors, comparisons)

        # Generate feedback
        feedback = self._generate_feedback(dimension, comparisons, score)

        return ReviewResult(
            dimension=dimension,
            score=score,
            comparisons=comparisons,
            feedback=feedback,
            loss=loss
        )

    def _compare_to_anchor(
        self,
        code: str,
        anchor: Anchor,
        dimension: ReviewDimension,
        focus: str
    ) -> Comparison:
        """
        Compare generated code to an anchor example.

        In production, this would call an LLM for comparison.
        Here we use heuristic scoring based on code characteristics.

        Args:
            code: Generated code
            anchor: Anchor to compare against
            dimension: Review dimension
            focus: Reviewer focus area

        Returns:
            Comparison result
        """
        # Heuristic comparison (in production: LLM call)
        code_score = self._heuristic_code_score(code, dimension)
        anchor_score = anchor.score10

        # Determine judgement
        diff = code_score - anchor_score
        if diff > 0.5:
            judgement = Judgement.BETTER
            confidence = min(0.9, 0.5 + abs(diff) * 0.1)
        elif diff < -0.5:
            judgement = Judgement.WORSE
            confidence = min(0.9, 0.5 + abs(diff) * 0.1)
        else:
            judgement = Judgement.TIE
            confidence = 0.6

        rationale = f"Compared to anchor score10: {anchor_score:.1f} on {focus}"

        return Comparison(
            anchor_id=anchor.id,
            anchor_score=anchor_score,
            judgement=judgement,
            confidence=confidence,
            rationale=rationale
        )

    def _heuristic_code_score(self, code: str, dimension: ReviewDimension) -> float:
        """
        Calculate heuristic score for code.

        Args:
            code: Code to score
            dimension: Dimension to evaluate

        Returns:
            Score from 1 to 10
        """
        score = 5.0  # Base score

        # Code quality indicators
        if dimension == ReviewDimension.CODE_QUALITY:
            if 'try:' in code and 'except' in code:
                score += 0.5  # Has error handling
            if 'def ' in code:
                score += 0.3  # Has functions
            if '"""' in code or "'''" in code:
                score += 0.3  # Has docstrings
            if len(code) > 100:
                score += 0.2  # Substantial code
            if 'raise' in code:
                score += 0.2  # Explicit exceptions

        # Interface consistency indicators
        elif dimension == ReviewDimension.INTERFACE_CONSISTENCY:
            if 'from typing import' in code or ': str' in code or ': int' in code:
                score += 0.5  # Type hints
            if '@dataclass' in code:
                score += 0.3  # Uses dataclasses
            if 'Optional[' in code:
                score += 0.2  # Handles optional types
            if '__init__' in code:
                score += 0.2  # Proper initialization

        # Pattern adherence indicators
        elif dimension == ReviewDimension.PATTERN_ADHERENCE:
            if 'class ' in code:
                score += 0.4  # Uses classes
            if 'self.' in code:
                score += 0.2  # Instance methods
            if 'import' in code:
                score += 0.2  # Modular imports
            if '_lock' in code or 'Lock' in code:
                score += 0.3  # Thread safety

        return min(10.0, max(1.0, score))

    def _compute_score_from_comparisons(
        self,
        anchors: List[Anchor],
        comparisons: List[Comparison]
    ) -> Tuple[float, float]:
        """
        Compute final score using weighted least squares fitting.

        Based on Idea2Paper's deterministic scoring algorithm:
        - Map judgement/confidence to probability
        - Grid search for score that minimizes loss

        Args:
            anchors: Anchor examples
            comparisons: Comparison results

        Returns:
            Tuple of (score, loss)
        """
        if not comparisons:
            return 5.0, 0.0

        # Build comparison map
        comp_map = {c.anchor_id: c for c in comparisons}

        probs: List[float] = []
        weights: List[float] = []
        scores: List[float] = []

        for anchor in anchors:
            comp = comp_map.get(anchor.id)
            if not comp:
                continue

            # Map judgement to probability
            confidence = max(0.0, min(1.0, comp.confidence))
            if comp.judgement == Judgement.BETTER:
                p = 0.5 + 0.45 * confidence
            elif comp.judgement == Judgement.WORSE:
                p = 0.5 - 0.45 * confidence
            else:
                p = 0.5

            probs.append(p)
            weights.append(anchor.weight)
            scores.append(anchor.score10)

        # Grid search for best score
        best_s = 5.0
        best_loss = float('inf')

        s = 1.0
        while s <= 10.0:
            loss = 0.0
            for p, w, anchor_score in zip(probs, weights, scores):
                pred = self._sigmoid(self.SIGMOID_K * (s - anchor_score))
                loss += w * (pred - p) ** 2

            if loss < best_loss:
                best_loss = loss
                best_s = s

            s += self.GRID_STEP

        return best_s, best_loss

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function."""
        return 1 / (1 + math.exp(-x))

    def _generate_feedback(
        self,
        dimension: ReviewDimension,
        comparisons: List[Comparison],
        score: float
    ) -> str:
        """Generate feedback text from comparisons."""
        better_count = sum(1 for c in comparisons if c.judgement == Judgement.BETTER)
        worse_count = sum(1 for c in comparisons if c.judgement == Judgement.WORSE)
        tie_count = sum(1 for c in comparisons if c.judgement == Judgement.TIE)

        return (
            f"{dimension.value}: Score {score:.1f}/10. "
            f"Compared to {len(comparisons)} anchors: "
            f"{better_count} better, {tie_count} tie, {worse_count} worse."
        )

    def _compute_final_result(
        self,
        reviews: List[ReviewResult],
        anchors: List[Anchor],
        context: Dict[str, Any]
    ) -> AnchoredReviewResult:
        """
        Compute final review result from all reviewers.

        Args:
            reviews: Individual review results
            anchors: Anchors used
            context: Generation context

        Returns:
            Complete AnchoredReviewResult
        """
        if not reviews:
            return self._neutral_result("No reviews completed")

        # Calculate average score
        scores = [r.score for r in reviews]
        avg_score = sum(scores) / len(scores)

        # Determine pass/fail (Scheme B: 2/3 >= q75 and avg >= q50)
        q50 = 5.0  # Median threshold
        q75 = 6.5  # Excellence threshold

        count_above_q75 = sum(1 for s in scores if s >= q75)
        passed = (count_above_q75 >= 2) and (avg_score >= q50)

        # Find main issue
        main_issue = None
        suggestions = []
        if not passed:
            min_score = min(scores)
            min_review = next(r for r in reviews if r.score == min_score)
            main_issue = min_review.dimension.value
            suggestions = self._generate_suggestions(min_review.dimension)

        # Build audit info
        audit = {
            'anchors': [{'id': a.id, 'score10': a.score10} for a in anchors],
            'dimension_scores': {r.dimension.value: r.score for r in reviews},
            'losses': {r.dimension.value: r.loss for r in reviews},
            'q50': q50,
            'q75': q75,
            'count_above_q75': count_above_q75
        }

        return AnchoredReviewResult(
            passed=passed,
            avg_score=avg_score,
            reviews=reviews,
            main_issue=main_issue,
            suggestions=suggestions,
            audit=audit
        )

    def _neutral_result(self, reason: str) -> AnchoredReviewResult:
        """Create neutral result when review cannot be performed."""
        return AnchoredReviewResult(
            passed=False,
            avg_score=5.0,
            reviews=[],
            main_issue="review_unavailable",
            suggestions=[reason],
            audit={'fallback': True, 'reason': reason}
        )

    def _generate_suggestions(self, dimension: ReviewDimension) -> List[str]:
        """Generate improvement suggestions for a dimension."""
        suggestions = {
            ReviewDimension.CODE_QUALITY: [
                "Add error handling with try/except blocks",
                "Include input validation",
                "Add docstrings to functions"
            ],
            ReviewDimension.INTERFACE_CONSISTENCY: [
                "Add type hints to function parameters",
                "Ensure consistent naming conventions",
                "Match expected API signatures"
            ],
            ReviewDimension.PATTERN_ADHERENCE: [
                "Follow established design patterns",
                "Use dependency injection where appropriate",
                "Maintain single responsibility principle"
            ]
        }
        return suggestions.get(dimension, ["Review code structure"])

    def _get_default_anchors(self, context: Dict[str, Any]) -> List[Anchor]:
        """Get default anchors when none provided."""
        # Default anchors representing different quality levels
        return [
            Anchor(id="anchor_low", name="Basic Implementation",
                   score10=4.0, weight=0.8),
            Anchor(id="anchor_mid", name="Standard Implementation",
                   score10=6.0, weight=1.0),
            Anchor(id="anchor_high", name="Best Practice Implementation",
                   score10=8.0, weight=1.2),
        ]
