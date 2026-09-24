from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from app.domain.evaluation import DimensionEvaluationStatus, EvaluationReasonCode
from app.services.evaluation.mpt_multimodal_adapter import (
    MultimodalEvaluatorAdapter,
    resolve_evaluator_version,
)


def test_resolve_evaluator_version():
    version = resolve_evaluator_version("openai", "gpt-4o")
    assert version == "openai/gpt-4o:evaluator-v1"


def test_evaluator_model_unavailable_when_unsupported():
    adapter = MultimodalEvaluatorAdapter(provider_id="claude_code")
    is_avail, reason = adapter.check_availability()
    assert is_avail is False
    assert "does not support multimodal" in reason

    # Calling evaluate returns structured ERROR
    res = adapter.evaluate_observation(
        prompt="evaluate this",
        image_paths=["dummy.jpg"],
    )
    assert res.status == DimensionEvaluationStatus.ERROR
    assert res.score is None
    assert res.error_code == EvaluationReasonCode.EVALUATOR_MODEL_UNAVAILABLE.value


def test_evaluator_success_with_custom_mock_client(tmp_path: Path):
    img_file = tmp_path / "img.png"
    Image.new("RGB", (50, 50), "red").save(str(img_file))

    mock_client = MagicMock(return_value="""
    {
      "status": "SCORED",
      "score": 0.88,
      "reason_codes": ["CLEAR_SUBJECT", "FAITHFUL"],
      "concise_summary": "Subject is clearly depicted with accurate visual style."
    }
    """)

    adapter = MultimodalEvaluatorAdapter(
        provider_id="openai",
        model_name="gpt-4o",
        custom_client=mock_client,
    )

    res = adapter.evaluate_observation(
        prompt="Assess visual clarity",
        image_paths=[str(img_file)],
    )

    assert res.status == DimensionEvaluationStatus.SCORED
    assert res.score == 0.88
    assert "CLEAR_SUBJECT" in res.reason_codes
    assert res.concise_summary is not None
    assert res.evaluator_version == "openai/gpt-4o:evaluator-v1"
    assert mock_client.call_count == 1


def test_evaluator_bounded_retry_on_schema_error(tmp_path: Path):
    img_file = tmp_path / "img.png"
    Image.new("RGB", (50, 50), "red").save(str(img_file))

    # First returns invalid text, second returns valid JSON
    mock_client = MagicMock(side_effect=[
        "Sorry I am an AI and here is my chain of thought...",
        '{"status": "SCORED", "score": 0.75, "reason_codes": ["RECOVERED"], "concise_summary": "Recovered."}',
    ])

    adapter = MultimodalEvaluatorAdapter(
        provider_id="openai",
        model_name="gpt-4o",
        custom_client=mock_client,
    )

    res = adapter.evaluate_observation(
        prompt="Assess visual clarity",
        image_paths=[str(img_file)],
    )

    assert res.status == DimensionEvaluationStatus.SCORED
    assert res.score == 0.75
    assert mock_client.call_count == 2


def test_evaluator_bounded_retry_exhausted(tmp_path: Path):
    img_file = tmp_path / "img.png"
    Image.new("RGB", (50, 50), "red").save(str(img_file))

    # Both calls return invalid text
    mock_client = MagicMock(side_effect=[
        "Invalid response 1",
        "Invalid response 2",
    ])

    adapter = MultimodalEvaluatorAdapter(
        provider_id="openai",
        model_name="gpt-4o",
        custom_client=mock_client,
    )

    res = adapter.evaluate_observation(
        prompt="Assess visual clarity",
        image_paths=[str(img_file)],
    )

    assert res.status == DimensionEvaluationStatus.ERROR
    assert res.score is None
    assert res.error_code == EvaluationReasonCode.EVALUATOR_SCHEMA_ERROR.value
    assert mock_client.call_count == 2
