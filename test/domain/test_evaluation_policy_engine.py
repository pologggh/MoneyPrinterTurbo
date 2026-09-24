import pytest

from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationDimensionMissingError,
)
from app.domain.evaluation_policy import (
    EvaluationPolicy,
    EvaluationPolicyEngine,
)


def _make_result(
    dim: EvaluationDimension,
    status: DimensionEvaluationStatus,
    score: float | None = None,
    target_id: str = "t-1",
) -> DimensionEvaluationResult:
    return DimensionEvaluationResult(
        evaluation_target_id=target_id,
        dimension=dim,
        status=status,
        score=score,
    )


def test_rule_1_critical_semantic_error_yields_indeterminate():
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.ERROR),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.9),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.9),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.9),
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.INDETERMINATE
    assert decision.overall_score is None
    assert "SEMANTIC_ALIGNMENT_ERROR" in decision.reason_codes


def test_rule_1_critical_knowledge_indeterminate_yields_indeterminate():
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.9),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.9),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.INDETERMINATE),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.9),
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.INDETERMINATE
    assert decision.overall_score is None
    assert "KNOWLEDGE_ACCURACY_INDETERMINATE" in decision.reason_codes


def test_rule_2_critical_semantic_hard_gate_below_threshold_fails():
    # Prompt explicit requirement:
    # semantic = 0.79, visual = 0.99, knowledge = 0.99, composition = 0.99 -> FAIL
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.79),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.99),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.99),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.99),
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.FAIL
    assert decision.overall_score is None
    assert "SEMANTIC_ALIGNMENT_BELOW_THRESHOLD" in decision.reason_codes


def test_rule_2_critical_knowledge_hard_gate_below_threshold_fails():
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.79),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.95),
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.FAIL
    assert decision.overall_score is None
    assert "KNOWLEDGE_ACCURACY_BELOW_THRESHOLD" in decision.reason_codes


def test_rule_3_non_critical_visual_error_yields_indeterminate():
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.ERROR),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.95),
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.INDETERMINATE
    assert decision.overall_score is None
    assert "VISUAL_QUALITY_ERROR" in decision.reason_codes


def test_rule_3_non_critical_composition_indeterminate_yields_indeterminate():
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.INDETERMINATE),
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.INDETERMINATE
    assert decision.overall_score is None
    assert "COMPOSITION_SUITABILITY_INDETERMINATE" in decision.reason_codes


def test_rule_4_non_critical_visual_below_threshold_fails():
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.64),  # < 0.65
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.95),
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.FAIL
    assert decision.overall_score is None
    assert "VISUAL_QUALITY_BELOW_THRESHOLD" in decision.reason_codes


def test_rule_4_non_critical_composition_below_threshold_fails():
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.64),  # < 0.65
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.FAIL
    assert decision.overall_score is None
    assert "COMPOSITION_SUITABILITY_BELOW_THRESHOLD" in decision.reason_codes


def test_rule_5_all_above_thresholds_but_weighted_below_minimum_fails():
    # Individual minimum thresholds: semantic=0.80, visual=0.65, knowledge=0.80, composition=0.65
    # Weighted total = 0.30*0.80 + 0.20*0.65 + 0.35*0.80 + 0.15*0.65 = 0.24 + 0.13 + 0.28 + 0.0975 = 0.7475 < 0.75
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.80),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.65),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.80),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.65),
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.FAIL
    assert decision.overall_score == 0.7475
    assert "OVERALL_SCORE_BELOW_THRESHOLD" in decision.reason_codes


def test_rule_5_all_pass_with_overall_score():
    # 0.30*0.90 + 0.20*0.85 + 0.35*0.95 + 0.15*0.80 = 0.27 + 0.17 + 0.3325 + 0.12 = 0.8925 >= 0.75
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.90),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.85),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.95),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.80),
    )
    decision = EvaluationPolicyEngine.evaluate(results)
    assert decision.decision == EvaluationDecision.PASS
    assert decision.overall_score == 0.8925
    assert "ALL_DIMENSIONS_PASSED" in decision.reason_codes


def test_policy_engine_requires_all_four_dimensions():
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.90),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.85),
    )
    with pytest.raises(EvaluationDimensionMissingError):
        EvaluationPolicyEngine.evaluate(results)


def test_policy_engine_custom_policy():
    custom_policy = EvaluationPolicy(
        policy_version="custom-v2",
        thresholds={
            EvaluationDimension.SEMANTIC_ALIGNMENT: 0.50,
            EvaluationDimension.VISUAL_QUALITY: 0.50,
            EvaluationDimension.KNOWLEDGE_ACCURACY: 0.50,
            EvaluationDimension.COMPOSITION_SUITABILITY: 0.50,
        },
        weights={
            EvaluationDimension.SEMANTIC_ALIGNMENT: 0.25,
            EvaluationDimension.VISUAL_QUALITY: 0.25,
            EvaluationDimension.KNOWLEDGE_ACCURACY: 0.25,
            EvaluationDimension.COMPOSITION_SUITABILITY: 0.25,
        },
        minimum_overall_score=0.60,
    )
    results = (
        _make_result(EvaluationDimension.SEMANTIC_ALIGNMENT, DimensionEvaluationStatus.SCORED, 0.65),
        _make_result(EvaluationDimension.VISUAL_QUALITY, DimensionEvaluationStatus.SCORED, 0.65),
        _make_result(EvaluationDimension.KNOWLEDGE_ACCURACY, DimensionEvaluationStatus.SCORED, 0.65),
        _make_result(EvaluationDimension.COMPOSITION_SUITABILITY, DimensionEvaluationStatus.SCORED, 0.65),
    )
    decision = EvaluationPolicyEngine.evaluate(results, policy=custom_policy)
    assert decision.decision == EvaluationDecision.PASS
    assert decision.policy_version == "custom-v2"
    assert decision.overall_score == 0.65
