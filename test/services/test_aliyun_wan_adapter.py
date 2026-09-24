import importlib.util
from pathlib import Path

import pytest

import app.services.asset_capability_registry as capability_registry
from app.config import config
from app.domain.asset_execution import ProviderOutcomeType, ProviderReceipt
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRoutingRequest,
    GenerationMode,
)
from app.domain.enums import VisualType
from app.services import aliyun_wan
from app.services.asset_adapters.aliyun_wan_adapter import AliyunWanAdapter
from app.services.asset_adapters.registry import AdapterRegistry


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class RecordingHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class FakeWanClient:
    def __init__(self, submit_result, status_results):
        self.submit_result = submit_result
        self.status_results = list(status_results)
        self.submit_calls = []
        self.status_calls = []

    def submit_video_task(self, **kwargs):
        self.submit_calls.append(kwargs)
        return self.submit_result

    def get_video_task(self, task_id):
        self.status_calls.append(task_id)
        return self.status_results.pop(0)


def make_request():
    return AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="revision-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=5,
        visual_goal="展示火箭升空",
        scene_description="清晨的发射场",
        generation_prompt="电影感，中国运载火箭升空",
        camera_movement="slow push in",
        aspect_ratio="16:9",
    )


def make_candidate():
    return AssetRouteCandidate(
        capability_id="aliyun_wan:wan2.7-t2v:TEXT_TO_VIDEO",
        provider="aliyun_wan",
        model="wan2.7-t2v",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        requested_visual_type=VisualType.AI_VIDEO,
        is_eligible=True,
    )


def test_adapter_registry_exposes_aliyun_wan_provider():
    registry = AdapterRegistry()

    assert registry.has_adapter("aliyun_wan")


def test_aliyun_wan_capability_provider_is_publicly_available():
    assert hasattr(capability_registry, "AliyunWanCapabilityAdapter")


def test_aliyun_wan_capability_requires_complete_configuration(monkeypatch):
    monkeypatch.setitem(config.app, "aliyun_wan_api_key", "test-key")
    monkeypatch.setitem(config.app, "aliyun_wan_workspace_id", "")
    provider = capability_registry.AliyunWanCapabilityAdapter()

    disabled_capability = provider.get_capabilities()[0]

    assert disabled_capability.enabled is False

    monkeypatch.setitem(config.app, "aliyun_wan_workspace_id", "ws-test")
    enabled_capability = provider.get_capabilities()[0]

    assert enabled_capability.enabled is True
    assert enabled_capability.provider == "aliyun_wan"
    assert enabled_capability.model == "wan2.7-t2v"
    assert enabled_capability.generation_mode == GenerationMode.TEXT_TO_VIDEO
    assert enabled_capability.supported_visual_types == (VisualType.AI_VIDEO,)
    assert enabled_capability.min_duration == 2
    assert enabled_capability.max_duration == 15


def test_default_capability_registry_lists_aliyun_wan():
    providers = {
        capability.provider
        for capability in capability_registry.get_default_capability_registry().list_capabilities()
    }

    assert "aliyun_wan" in providers


def test_example_config_documents_aliyun_wan_settings():
    example_config = Path("config.example.toml").read_text(encoding="utf-8")

    for setting_name in (
        "aliyun_wan_api_key",
        "aliyun_wan_workspace_id",
        "aliyun_wan_region",
        "aliyun_wan_model",
        "aliyun_wan_resolution",
        "aliyun_wan_poll_interval",
        "aliyun_wan_run_timeout",
    ):
        assert setting_name in example_config


def test_aliyun_wan_client_module_is_available():
    assert importlib.util.find_spec("app.services.aliyun_wan") is not None


def test_aliyun_wan_client_is_publicly_available():
    assert hasattr(aliyun_wan, "AliyunWanClient")


def test_aliyun_wan_is_enabled_only_with_key_and_workspace():
    assert not aliyun_wan.is_enabled(
        {"aliyun_wan_api_key": "key-only", "aliyun_wan_workspace_id": ""}
    )
    assert aliyun_wan.is_enabled(
        {
            "aliyun_wan_api_key": "test-key",
            "aliyun_wan_workspace_id": "ws-test",
        }
    )
    assert not aliyun_wan.is_enabled(
        {
            "aliyun_wan_api_key": "test-key",
            "aliyun_wan_workspace_id": "ws-test",
            "aliyun_wan_region": "unsupported-region",
        }
    )
    assert not aliyun_wan.is_enabled(
        {
            "aliyun_wan_api_key": "test-key",
            "aliyun_wan_workspace_id": "ws-test",
            "aliyun_wan_model": "wan-image-to-video",
        }
    )


def test_submit_video_task_uses_official_async_endpoint_and_payload():
    http = RecordingHttpClient(
        FakeResponse(
            {
                "output": {"task_id": "task-123", "task_status": "PENDING"},
                "request_id": "request-456",
            }
        )
    )
    client = aliyun_wan.AliyunWanClient(
        settings={
            "aliyun_wan_api_key": "test-key",
            "aliyun_wan_workspace_id": "ws-test",
            "aliyun_wan_region": "cn-beijing",
            "aliyun_wan_model": "wan2.7-t2v",
            "aliyun_wan_resolution": "720P",
            "aliyun_wan_prompt_extend": True,
            "aliyun_wan_watermark": False,
        },
        http_client=http,
    )

    result = client.submit_video_task(
        prompt="中国航天发展史",
        duration=5,
        ratio="16:9",
    )

    assert result == {
        "task_id": "task-123",
        "status": "PENDING",
        "request_id": "request-456",
    }
    assert http.calls == [
        (
            "https://ws-test.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
            {
                "headers": {
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                    "X-DashScope-Async": "enable",
                },
                "json": {
                    "model": "wan2.7-t2v",
                    "input": {"prompt": "中国航天发展史"},
                    "parameters": {
                        "resolution": "720P",
                        "ratio": "16:9",
                        "prompt_extend": True,
                        "watermark": False,
                        "duration": 5,
                    },
                },
                "timeout": 30,
            },
        )
    ]


def test_get_video_task_normalizes_success_response():
    http = RecordingHttpClient(
        FakeResponse(
            {
                "output": {
                    "task_id": "task-123",
                    "task_status": "SUCCEEDED",
                    "video_url": "https://example.test/video.mp4",
                },
                "request_id": "request-789",
            }
        )
    )
    client = aliyun_wan.AliyunWanClient(
        settings={
            "aliyun_wan_api_key": "test-key",
            "aliyun_wan_workspace_id": "ws-test",
            "aliyun_wan_region": "cn-beijing",
        },
        http_client=http,
    )

    result = client.get_video_task("task-123")

    assert result == {
        "task_id": "task-123",
        "status": "SUCCEEDED",
        "video_url": "https://example.test/video.mp4",
        "request_id": "request-789",
        "error_code": "",
        "error_message": "",
    }
    assert http.calls == [
        (
            "https://ws-test.cn-beijing.maas.aliyuncs.com/api/v1/tasks/task-123",
            {
                "headers": {
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                    "X-DashScope-Async": "enable",
                },
                "timeout": 30,
            },
        )
    ]


def test_submit_rejects_unsupported_duration_before_http_call():
    http = RecordingHttpClient(FakeResponse({}))
    client = aliyun_wan.AliyunWanClient(
        settings={
            "aliyun_wan_api_key": "test-key",
            "aliyun_wan_workspace_id": "ws-test",
        },
        http_client=http,
    )

    with pytest.raises(ValueError, match="between 2 and 15"):
        client.submit_video_task(prompt="test", duration=16, ratio="16:9")

    assert http.calls == []


def test_submit_rejects_incomplete_configuration_before_http_call():
    http = RecordingHttpClient(FakeResponse({}))
    client = aliyun_wan.AliyunWanClient(
        settings={"aliyun_wan_api_key": "test-key", "aliyun_wan_workspace_id": ""},
        http_client=http,
    )

    with pytest.raises(aliyun_wan.AliyunWanConfigurationError, match="workspace"):
        client.submit_video_task(prompt="test", duration=5, ratio="16:9")

    assert http.calls == []


def test_submit_rejects_unsupported_phase_one_model_before_http_call():
    http = RecordingHttpClient(FakeResponse({}))
    client = aliyun_wan.AliyunWanClient(
        settings={
            "aliyun_wan_api_key": "test-key",
            "aliyun_wan_workspace_id": "ws-test",
            "aliyun_wan_model": "wan-image-to-video",
        },
        http_client=http,
    )

    with pytest.raises(aliyun_wan.AliyunWanConfigurationError, match="wan2.7-t2v"):
        client.submit_video_task(prompt="test", duration=5, ratio="16:9")

    assert http.calls == []


def test_submit_classifies_http_400_as_definitive_rejection():
    http = RecordingHttpClient(FakeResponse({"code": "InvalidParameter"}, 400))
    client = aliyun_wan.AliyunWanClient(
        settings={
            "aliyun_wan_api_key": "test-key",
            "aliyun_wan_workspace_id": "ws-test",
        },
        http_client=http,
    )

    with pytest.raises(aliyun_wan.AliyunWanRequestRejectedError):
        client.submit_video_task(prompt="test", duration=5, ratio="16:9")


def test_adapter_executes_wan_task_and_returns_video_url():
    client = FakeWanClient(
        submit_result={
            "task_id": "task-123",
            "status": "PENDING",
            "request_id": "request-1",
        },
        status_results=[
            {
                "task_id": "task-123",
                "status": "SUCCEEDED",
                "video_url": "https://example.test/video.mp4",
                "request_id": "request-2",
                "error_code": "",
                "error_message": "",
            }
        ],
    )
    adapter = AliyunWanAdapter(client=client, poll_interval=0)

    result = adapter.execute(make_request(), make_candidate(), Path("."))

    assert client.submit_calls == [
        {
            "prompt": "电影感，中国运载火箭升空",
            "duration": 5,
            "ratio": "16:9",
        }
    ]
    assert client.status_calls == ["task-123"]
    assert result.outcome_type == ProviderOutcomeType.SUCCESS
    assert result.status == "SUCCEEDED"
    assert result.remote_task_id == "task-123"
    assert result.file_path == "https://example.test/video.mp4"


def test_adapter_submit_returns_receipt_before_status_polling():
    client = FakeWanClient(
        submit_result={
            "task_id": "task-durable",
            "status": "PENDING",
            "request_id": "request-durable",
        },
        status_results=[],
    )
    adapter = AliyunWanAdapter(client=client, poll_interval=0)

    receipt = adapter.submit(make_request(), make_candidate(), Path("."))

    assert isinstance(receipt, ProviderReceipt)
    assert receipt.provider == "aliyun_wan"
    assert receipt.provider_job_id == "task-durable"
    assert receipt.provider_status == "PENDING"
    assert receipt.sanitized_metadata == {"request_id": "request-durable"}
    assert client.status_calls == []


def test_get_status_polls_existing_task_until_terminal_success():
    client = FakeWanClient(
        submit_result={},
        status_results=[
            {
                "task_id": "task-existing",
                "status": "PENDING",
                "video_url": "",
                "request_id": "request-1",
                "error_code": "",
                "error_message": "",
            },
            {
                "task_id": "task-existing",
                "status": "RUNNING",
                "video_url": "",
                "request_id": "request-2",
                "error_code": "",
                "error_message": "",
            },
            {
                "task_id": "task-existing",
                "status": "SUCCEEDED",
                "video_url": "https://example.test/final.mp4",
                "request_id": "request-3",
                "error_code": "",
                "error_message": "",
            },
        ],
    )
    adapter = AliyunWanAdapter(client=client, poll_interval=0, max_polls=3)

    result = adapter.get_status("task-existing", Path("."))

    assert client.status_calls == ["task-existing"] * 3
    assert result.outcome_type == ProviderOutcomeType.SUCCESS
    assert result.status == "SUCCEEDED"
    assert result.file_path == "https://example.test/final.mp4"


def test_get_status_timeout_preserves_remote_task_id_for_recovery():
    running_snapshot = {
        "task_id": "task-still-running",
        "status": "RUNNING",
        "video_url": "",
        "request_id": "request-running",
        "error_code": "",
        "error_message": "",
    }
    client = FakeWanClient(
        submit_result={},
        status_results=[running_snapshot, running_snapshot],
    )
    adapter = AliyunWanAdapter(client=client, poll_interval=0, max_polls=2)

    result = adapter.get_status("task-still-running", Path("."))

    assert client.status_calls == ["task-still-running"] * 2
    assert result.outcome_type == ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN
    assert result.status == "RUNNING"
    assert result.remote_task_id == "task-still-running"
    assert result.error_code == "POLL_TIMEOUT"


def test_adapter_treats_submission_transport_failure_as_ambiguous():
    class FailingSubmitClient:
        def submit_video_task(self, **kwargs):
            raise ConnectionError("connection closed after request was sent")

    adapter = AliyunWanAdapter(client=FailingSubmitClient(), poll_interval=0)

    result = adapter.execute(make_request(), make_candidate(), Path("."))

    assert result.outcome_type == ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN
    assert result.error_code == "SUBMISSION_OUTCOME_UNKNOWN"


def test_adapter_maps_configuration_error_to_capability_incompatible():
    class MisconfiguredClient:
        def submit_video_task(self, **kwargs):
            raise aliyun_wan.AliyunWanConfigurationError(
                "Aliyun Wan workspace ID is not configured"
            )

    adapter = AliyunWanAdapter(client=MisconfiguredClient(), poll_interval=0)

    result = adapter.execute(make_request(), make_candidate(), Path("."))

    assert result.outcome_type == ProviderOutcomeType.CAPABILITY_INCOMPATIBLE
    assert result.error_code == "PROVIDER_NOT_CONFIGURED"


def test_adapter_maps_provider_rejection_to_definitive_failure():
    class RejectedClient:
        def submit_video_task(self, **kwargs):
            raise aliyun_wan.AliyunWanRequestRejectedError(
                "Wan task submission was rejected with HTTP 400"
            )

    adapter = AliyunWanAdapter(client=RejectedClient(), poll_interval=0)

    result = adapter.execute(make_request(), make_candidate(), Path("."))

    assert result.outcome_type == ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE
    assert result.error_code == "SUBMISSION_REJECTED"


def test_adapter_maps_terminal_provider_failure_without_fallback_resubmit():
    client = FakeWanClient(
        submit_result={
            "task_id": "task-failed",
            "status": "PENDING",
            "request_id": "request-1",
        },
        status_results=[
            {
                "task_id": "task-failed",
                "status": "FAILED",
                "video_url": "",
                "request_id": "request-2",
                "error_code": "InvalidParameter",
                "error_message": "prompt rejected",
            }
        ],
    )
    adapter = AliyunWanAdapter(client=client, poll_interval=0)

    result = adapter.execute(make_request(), make_candidate(), Path("."))

    assert result.outcome_type == ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE
    assert result.status == "FAILED"
    assert result.remote_task_id == "task-failed"
    assert result.error_code == "InvalidParameter"
    assert len(client.submit_calls) == 1
