from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from app.domain.asset_execution import ProviderOutcomeType
from app.domain.asset_router import (
    AssetCapability,
    GenerationMode,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.content_plan import ContentBeat
from app.domain.enums import VisualType
from app.domain.evaluation import DimensionEvaluationStatus
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard_agent import (
    StoryboardExecutionResult,
    StoryboardPlanningInput,
)
from app.services.asset_adapters.base import (
    AdapterExecutionResult,
    AssetExecutionAdapter,
)
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.evaluation.mpt_multimodal_adapter import (
    MultimodalEvaluationObserverResult,
    MultimodalEvaluatorAdapter,
)


def create_deterministic_offline_planner_llm(topic: str, target_duration: float, evidence_id: str):
    """Creates a deterministic offline planner LLM caller returning valid JSON beats."""
    def _caller(prompt: str) -> str:
        dur = round(target_duration / 2.0, 1)
        return json.dumps({
            "title": f"Explaining {topic}",
            "beats": [
                {
                    "planner_ref": "b1",
                    "beat_type": "HOOK",
                    "intent": f"Introduce {topic} with an intriguing hook",
                    "order": 1,
                    "target_duration": dur,
                    "importance": 0.8,
                    "evidence_refs": [evidence_id],
                },
                {
                    "planner_ref": "b2",
                    "beat_type": "KNOWLEDGE",
                    "intent": f"Core explanation of {topic}",
                    "order": 2,
                    "target_duration": dur,
                    "importance": 0.95,
                    "evidence_refs": [evidence_id],
                },
            ],
        })
    return _caller


class DeterministicOfflineStoryboardAgent:
    """Deterministic offline StoryboardAgent generating 1 shot revision per beat."""

    def generate_shots_for_beat(
        self,
        input_data: Any,
        shot_repository: Any | None = None,
        **kwargs: Any,
    ) -> StoryboardExecutionResult:
        if isinstance(input_data, StoryboardPlanningInput):
            beat = input_data.beat
            topic = input_data.video_title or ""
        elif isinstance(input_data, ContentBeat):
            beat = input_data
            topic = kwargs.get("topic", "")
        else:
            beat = getattr(input_data, "beat", input_data)
            topic = getattr(input_data, "video_title", "")

        shot_id = f"shot_{beat.beat_id[:8]}"
        shot = Shot(
            shot_id=shot_id,
            beat_lineage_id=beat.beat_lineage_id,
            local_order=1,
        )
        shot_rev = ShotRevision(
            shot_revision_id=f"srev_{beat.beat_id[:8]}",
            shot_id=shot_id,
            revision_number=1,
            beat_lineage_id=beat.beat_lineage_id,
            created_from_beat_instance_id=beat.beat_id,
            narration=f"Exploring {beat.intent}",
            visual_goal=f"Clear diagram illustrating {beat.intent}",
            scene_description=f"Detailed educational scene for {beat.intent}",
            generation_prompt=f"Educational diagram illustrating {topic or beat.intent}, high resolution",
            visual_type=VisualType.AI_IMAGE,
            target_duration=beat.target_duration,
            camera_movement="static",
            evidence_refs=tuple(beat.evidence_refs),
        )
        return StoryboardExecutionResult(
            shots=(shot,),
            shot_revisions=(shot_rev,),
        )


class DeterministicOfflineAssetAdapter(AssetExecutionAdapter):
    """
    Offline asset adapter that generates valid deterministic PNG images without network calls.
    """

    def __init__(self, provider_name: str = "offline_image_provider", model_name: str = "offline_v1"):
        self.provider_name = provider_name
        self.model_name = model_name

    def execute(
        self,
        request: Any,
        candidate: Any,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        out_path = Path(target_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = f"asset_{request.shot_id}_{uuid4().hex[:8]}.png"
        img_file = out_path / filename

        # Create a valid test image
        img = Image.new("RGB", (640, 360), color=(60, 100, 160))
        img.save(img_file)

        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="SUCCEEDED",
            file_path=str(img_file),
            raw_response={"provider": self.provider_name, "model": self.model_name},
        )

    def submit(
        self,
        request: Any,
        candidate: Any,
        output_dir: str | Path,
        idempotency_key: str | None = None,
    ) -> AdapterExecutionResult:
        return self.execute(request, candidate, Path(output_dir))

    def poll(self, job_id: str) -> AdapterExecutionResult:
        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="SUCCEEDED",
        )


class DeterministicOfflineEvaluatorAdapter(MultimodalEvaluatorAdapter):
    """
    Offline multimodal evaluator adapter that returns deterministic scores.
    Can be configured to fail specific shots or dimensions for testing quality remediation.
    """

    def __init__(
        self,
        failing_shot_tokens: set[str] | None = None,
        evaluator_version: str = "offline-eval-v1",
    ):
        self.failing_shot_tokens = failing_shot_tokens or set()
        self.evaluator_version = evaluator_version

    def evaluate_observation(
        self,
        prompt: str,
        image_paths: tuple[str, ...] | list[str],
    ) -> MultimodalEvaluationObserverResult:
        # Check if any image filename or prompt text contains a failing token
        is_failing = any(
            token.lower() in Path(p).name.lower()
            for token in self.failing_shot_tokens
            for p in image_paths
        ) or any(
            token.lower() in prompt.lower()
            for token in self.failing_shot_tokens
        )
        if is_failing:
            return MultimodalEvaluationObserverResult(
                status=DimensionEvaluationStatus.SCORED,
                score=0.45,
                reason_codes=("SIMULATED_DEFECT",),
                concise_summary="Simulated offline evaluation defect",
                evaluator_version=self.evaluator_version,
            )

        return MultimodalEvaluationObserverResult(
            status=DimensionEvaluationStatus.SCORED,
            score=0.92,
            reason_codes=("HIGH_QUALITY",),
            concise_summary="Accurate and sharp presentation",
            evaluator_version=self.evaluator_version,
        )


def build_offline_benchmark_environment(storage_dir: Path, failing_shot_tokens: set[str] | None = None):
    """
    Builds the full suite of offline adapters and capabilities for running benchmarks in OFFLINE mode.
    Zero external/network calls.
    """
    adapter = DeterministicOfflineAssetAdapter(provider_name="offline_provider", model_name="offline_model")
    adapter_registry = AdapterRegistry()
    adapter_registry.register_adapter("offline_provider", adapter)

    eval_adapter = DeterministicOfflineEvaluatorAdapter(
        failing_shot_tokens=failing_shot_tokens,
        evaluator_version="offline-eval-v1",
    )

    cap = AssetCapability(
        capability_id="cap-offline-1",
        provider="offline_provider",
        model="offline_model",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
        supported_visual_types=(VisualType.AI_IMAGE,),
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.LOW,
            latency_tier=TierLevel.LOW,
        ),
    )

    return {
        "adapter_registry": adapter_registry,
        "evaluator_adapter": eval_adapter,
        "capabilities": (cap,),
        "storyboard_agent": DeterministicOfflineStoryboardAgent(),
    }
