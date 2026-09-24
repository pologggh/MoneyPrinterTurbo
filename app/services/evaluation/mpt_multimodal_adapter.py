from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from app.config import config
from app.domain.evaluation import DimensionEvaluationStatus, EvaluationReasonCode
from app.models.llm_provider import get_llm_provider

EVALUATOR_SUFFIX: str = ":evaluator-v1"
MAX_EVALUATOR_ATTEMPTS: int = 2

EVALUATION_SYSTEM_PROMPT = """
You are an objective multimodal video and image evaluation observer.
Your role is to strictly observe and assess visual media against specified criteria.
You do NOT make final pass/fail decisions for the application.
You must return a raw JSON object with no markdown formatting, no code blocks, and no chain of thought.

Schema:
{
  "status": "SCORED" | "INDETERMINATE" | "ERROR",
  "score": <float between 0.0 and 1.0 if SCORED, else null>,
  "reason_codes": [<string>, ...],
  "concise_summary": "<factual 1-2 sentence summary of visual observation>"
}
""".strip()


class MultimodalEvaluationObserverResult(BaseModel):
    """
    Structured outcome of a multimodal model evaluation request.
    Strictly observes without making final PASS/FAIL policy decisions.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: DimensionEvaluationStatus
    score: float | None = None
    reason_codes: tuple[str, ...] = ()
    concise_summary: str | None = None
    error_code: str | None = None
    evaluator_version: str


def resolve_evaluator_version(
    provider_id: str | None = None,
    model_name: str | None = None,
) -> str:
    """Resolves canonical evaluator version string from MPT configuration."""
    pid = (provider_id or config.app.get("llm_provider", "moonshot")).lower()
    spec = get_llm_provider(pid)
    resolved_model = model_name
    if not resolved_model:
        if spec:
            resolved_model = spec.resolve_model_name(config.app.get(spec.config_key("model_name"), ""))
        else:
            resolved_model = "unknown"
    return f"{pid}/{resolved_model}{EVALUATOR_SUFFIX}"


def _encode_image_to_data_url(image_path: str) -> str:
    """Reads a local image file and converts to base64 data URL."""
    p = Path(image_path)
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/jpeg"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _parse_and_validate_observer_json(
    raw_text: str,
    evaluator_version: str,
) -> MultimodalEvaluationObserverResult | None:
    """Extracts, parses, and validates the observer JSON output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Try finding JSON object in text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None

    if not isinstance(data, dict):
        return None

    raw_status = str(data.get("status", "")).upper()
    if raw_status not in {s.value for s in DimensionEvaluationStatus}:
        return None

    status = DimensionEvaluationStatus(raw_status)
    raw_score = data.get("score")
    score: float | None = None

    if status == DimensionEvaluationStatus.SCORED:
        if raw_score is None:
            return None
        try:
            score_val = float(raw_score)
            if not (0.0 <= score_val <= 1.0):
                return None
            score = round(score_val, 4)
        except (ValueError, TypeError):
            return None
    else:
        score = None

    reason_codes_list = data.get("reason_codes", [])
    if isinstance(reason_codes_list, list):
        reason_codes = tuple(str(c) for c in reason_codes_list if c)
    else:
        reason_codes = ()

    concise_summary = data.get("concise_summary")
    if concise_summary is not None:
        concise_summary = str(concise_summary).strip()

    return MultimodalEvaluationObserverResult(
        status=status,
        score=score,
        reason_codes=reason_codes,
        concise_summary=concise_summary,
        evaluator_version=evaluator_version,
    )


class MultimodalEvaluatorAdapter:
    """
    Adapter reusing MoneyPrinterTurbo's configured LLM/multimodal infrastructure
    without maintaining a duplicate model registry.
    """

    def __init__(
        self,
        provider_id: str | None = None,
        model_name: str | None = None,
        custom_client: Any | None = None,
    ):
        self._provider_id = (provider_id or config.app.get("llm_provider", "moonshot")).lower()
        self._provider_spec = get_llm_provider(self._provider_id)
        self._custom_client = custom_client

        if self._provider_spec:
            self._model_name = (
                model_name
                or self._provider_spec.resolve_model_name(
                    config.app.get(self._provider_spec.config_key("model_name"), "")
                )
            )
            self._base_url = self._provider_spec.resolve_base_url(
                config.app.get(self._provider_spec.config_key("base_url"), "")
            )
            self._api_key = config.app.get(self._provider_spec.config_key("api_key"), "")
        else:
            self._model_name = model_name or "unknown"
            self._base_url = ""
            self._api_key = ""

        self.evaluator_version = f"{self._provider_id}/{self._model_name}{EVALUATOR_SUFFIX}"

    def check_availability(self) -> tuple[bool, str | None]:
        """
        Verifies whether current MPT configuration can fulfill multimodal evaluation.
        """
        if not self._provider_spec:
            return False, f"Unknown provider '{self._provider_id}'"

        if self._provider_spec.adapter not in {
            "openai_compatible",
            "azure",
            "gemini",
            "ollama",
            "dashscope",
            "litellm",
        }:
            return False, f"Provider '{self._provider_id}' adapter '{self._provider_spec.adapter}' does not support multimodal evaluation"

        if self._provider_spec.requires_api_key and not self._api_key and not self._custom_client:
            return False, f"Missing API key for provider '{self._provider_id}'"

        return True, None

    def evaluate_observation(
        self,
        prompt: str,
        image_paths: Sequence[str],
        system_prompt: str = EVALUATION_SYSTEM_PROMPT,
    ) -> MultimodalEvaluationObserverResult:
        """
        Sends multimodal evaluation prompt and images to configured model with bounded retries.
        """
        is_avail, reason = self.check_availability()
        if not is_avail:
            return MultimodalEvaluationObserverResult(
                status=DimensionEvaluationStatus.ERROR,
                score=None,
                reason_codes=(EvaluationReasonCode.EVALUATOR_MODEL_UNAVAILABLE.value,),
                error_code=EvaluationReasonCode.EVALUATOR_MODEL_UNAVAILABLE.value,
                concise_summary=reason,
                evaluator_version=self.evaluator_version,
            )

        # Validate that image paths exist
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_paths:
            return MultimodalEvaluationObserverResult(
                status=DimensionEvaluationStatus.ERROR,
                score=None,
                reason_codes=(EvaluationReasonCode.EVALUATION_MEDIA_UNREADABLE.value,),
                error_code=EvaluationReasonCode.EVALUATION_MEDIA_UNREADABLE.value,
                concise_summary="No valid image files provided for evaluation",
                evaluator_version=self.evaluator_version,
            )

        last_error = ""
        is_schema_error = False

        for attempt in range(1, MAX_EVALUATOR_ATTEMPTS + 1):
            try:
                raw_text = self._invoke_model(prompt, valid_paths, system_prompt)
                parsed = _parse_and_validate_observer_json(raw_text, self.evaluator_version)
                if parsed is not None:
                    return parsed

                is_schema_error = True
                last_error = f"Invalid observer JSON format: {raw_text[:200]}"
                logger.warning(
                    f"Multimodal evaluator schema error on attempt {attempt}/{MAX_EVALUATOR_ATTEMPTS}: {last_error}"
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning(
                    f"Multimodal evaluator call failed on attempt {attempt}/{MAX_EVALUATOR_ATTEMPTS}: {exc}"
                )

        error_code = (
            EvaluationReasonCode.EVALUATOR_SCHEMA_ERROR.value
            if is_schema_error
            else EvaluationReasonCode.EVALUATOR_CALL_FAILED.value
        )
        return MultimodalEvaluationObserverResult(
            status=DimensionEvaluationStatus.ERROR,
            score=None,
            reason_codes=(error_code,),
            error_code=error_code,
            concise_summary=f"Evaluation failed after {MAX_EVALUATOR_ATTEMPTS} attempts: {last_error[:150]}",
            evaluator_version=self.evaluator_version,
        )

    def _invoke_model(
        self,
        prompt: str,
        image_paths: Sequence[str],
        system_prompt: str,
    ) -> str:
        """Invokes the model client with multimodal content."""
        if self._custom_client is not None:
            # For testing: custom_client can be a callable or an object
            if callable(self._custom_client):
                return self._custom_client(prompt, image_paths, system_prompt)
            if hasattr(self._custom_client, "evaluate"):
                return self._custom_client.evaluate(prompt, image_paths, system_prompt)

        # Build message payload for OpenAI-compatible endpoint
        content_items: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            data_url = _encode_image_to_data_url(path)
            content_items.append({"type": "image_url", "image_url": {"url": data_url}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_items},
        ]

        if self._provider_spec and self._provider_spec.adapter == "azure":
            from openai import AzureOpenAI

            api_version = config.app.get(self._provider_spec.config_key("api_version"), "2024-02-01")
            client = AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._base_url,
                api_version=api_version,
            )
        else:
            from openai import OpenAI

            client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url or None,
            )

        response = client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )

        choice = response.choices[0]
        return choice.message.content or ""
