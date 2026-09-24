from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from app.controllers.v1.storyboard import (
    ApiCreateContentPlanRequest,
    ApiCreateStoryboardFromTopicRequest,
    create_content_plan,
    create_storyboard_from_topic,
)
from app.domain.planner import (
    ContentPlanner,
    InsufficientEvidenceError,
    PlannerInput,
)
from app.services.storyboard_generation_pipeline import (
    StoryboardGenerationInput,
    StoryboardGenerationPipeline,
)


def _make_llm_json(beats: list[dict]) -> str:
    return json.dumps({
        "title": "Quantum Principles",
        "beats": beats,
        "planner_notes": "notes",
    })


def test_source_grounded_rejects_missing_evidence_input():
    valid_beats = _make_llm_json([
        {
            "planner_ref": "b1",
            "beat_type": "KNOWLEDGE",
            "intent": "Explain superposition",
            "order": 1,
            "target_duration": 30.0,
            "importance": 0.8,
            "evidence_refs": ["ev_doc_1"],
            "inherit_from": None,
        }
    ])
    planner = ContentPlanner(llm_caller=lambda p: valid_beats)
    inp = PlannerInput(
        topic="Quantum Computing",
        target_video_duration=30.0,
        source_grounded=True,
        knowledge_context=(),
        available_evidence_ids=(),
    )
    with pytest.raises(InsufficientEvidenceError) as exc_info:
        planner.plan(inp)
    assert "no available evidence IDs" in str(exc_info.value)


def test_source_grounded_rejects_knowledge_beat_without_refs():
    """Area 5: When source_grounded=True, KNOWLEDGE beat without evidence_refs is rejected."""
    llm_output = _make_llm_json([
        {
            "planner_ref": "b1",
            "beat_type": "KNOWLEDGE",
            "intent": "Explain superposition",
            "order": 1,
            "target_duration": 30.0,
            "importance": 1.0,
            "evidence_refs": [],  # Missing required evidence
            "inherit_from": None,
        }
    ])
    planner = ContentPlanner(llm_caller=lambda p: llm_output)
    inp = PlannerInput(
        topic="Quantum Computing",
        target_video_duration=30.0,
        source_grounded=True,
        available_evidence_ids=("ev_valid_01",),
    )
    with pytest.raises(InsufficientEvidenceError) as exc_info:
        planner.plan(inp)
    assert "requires evidence references" in str(exc_info.value)


def test_source_grounded_rejects_hallucinated_evidence_id():
    """Area 5: When source_grounded=True, evidence refs not in available_evidence_ids are rejected."""
    llm_output = _make_llm_json([
        {
            "planner_ref": "b1",
            "beat_type": "KNOWLEDGE",
            "intent": "Explain superposition",
            "order": 1,
            "target_duration": 30.0,
            "importance": 1.0,
            "evidence_refs": ["ev_hallucinated_999"],  # Hallucinated ID
            "inherit_from": None,
        }
    ])
    planner = ContentPlanner(llm_caller=lambda p: llm_output)
    inp = PlannerInput(
        topic="Quantum Computing",
        target_video_duration=30.0,
        source_grounded=True,
        available_evidence_ids=("ev_valid_01", "ev_valid_02"),
    )
    with pytest.raises(InsufficientEvidenceError) as exc_info:
        planner.plan(inp)
    assert "is not grounded in available evidence IDs" in str(exc_info.value)
    assert "ev_hallucinated_999" in str(exc_info.value)


def test_source_grounded_accepts_valid_grounded_beats():
    """Area 5: Valid evidence refs matching declared evidence IDs pass cleanly."""
    llm_output = _make_llm_json([
        {
            "planner_ref": "b1",
            "beat_type": "HOOK",
            "intent": "Intro",
            "order": 1,
            "target_duration": 10.0,
            "importance": 0.8,
            "evidence_refs": [],
            "inherit_from": None,
        },
        {
            "planner_ref": "b2",
            "beat_type": "KNOWLEDGE",
            "intent": "Explain superposition",
            "order": 2,
            "target_duration": 20.0,
            "importance": 1.0,
            "evidence_refs": ["ev_valid_01"],
            "inherit_from": None,
        },
    ])
    planner = ContentPlanner(llm_caller=lambda p: llm_output)
    inp = PlannerInput(
        topic="Quantum Computing",
        target_video_duration=30.0,
        source_grounded=True,
        available_evidence_ids=("ev_valid_01",),
    )
    plan = planner.plan(inp)
    assert len(plan.beats) == 2
    assert plan.beats[1].evidence_refs == ("ev_valid_01",)


def test_pipeline_rejects_source_grounded_without_evidence_items():
    """Area 5: StoryboardGenerationPipeline raises ValueError if source_grounded=True without evidence."""
    pipeline = StoryboardGenerationPipeline(llm_caller=lambda p: "")
    inp = StoryboardGenerationInput(
        topic="Quantum",
        source_grounded=True,
        evidence_items=(),
        knowledge_context=(),
    )
    with pytest.raises(ValueError, match="requires non-empty evidence_items"):
        pipeline.generate(inp)


def test_api_controller_rejects_source_grounded_without_evidence():
    """Area 5: API endpoints return 400 when source_grounded=True without evidence."""
    req_storyboard = ApiCreateStoryboardFromTopicRequest(
        topic="Black Holes",
        source_grounded=True,
        evidence_items=[],
        knowledge_context=[],
    )
    with pytest.raises(HTTPException) as exc_info:
        create_storyboard_from_topic(req_storyboard)
    assert exc_info.value.status_code == 400
    assert "requires non-empty evidence_items" in exc_info.value.detail

    req_plan = ApiCreateContentPlanRequest(
        topic="Black Holes",
        source_grounded=True,
        evidence_items=[],
        knowledge_context=[],
    )
    with pytest.raises(HTTPException) as exc_info2:
        create_content_plan(req_plan)
    assert exc_info2.value.status_code == 400
    assert "requires non-empty evidence_items" in exc_info2.value.detail
