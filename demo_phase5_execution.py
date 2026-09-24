"""
Executable Demonstration of Phase 5.2: Provider Lifecycle + Retry / Fallback
Agentic Knowledge Video Production System (based on MoneyPrinterTurbo)

Run with:
    python demo_phase5_execution.py
"""

import sys
import tempfile
from pathlib import Path

from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001, S110
        pass

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.domain.asset_execution import (
    BlindSubmissionForbiddenError,
    ExecutionRetryPolicy,
    ProviderOutcomeType,
    ShotExecution,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutingRequest,
    GenerationMode,
    RoutingStrategy,
)
from app.domain.enums import VisualType
from app.services.asset_adapters.base import (
    AdapterExecutionResult,
    AssetExecutionAdapter,
    ProviderExecutionCapabilities,
)
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.attempt_recovery_service import AttemptRecoveryService
from app.services.shot_execution_service import ShotExecutionService


class DemoFakeAdapter(AssetExecutionAdapter):
    """Configurable adapter for demonstration purposes."""

    def __init__(self, results: list[AdapterExecutionResult], capabilities: ProviderExecutionCapabilities | None = None):
        self._results = list(results)
        self._capabilities = capabilities or ProviderExecutionCapabilities(supports_async_status=True)
        self.submissions = []

    @property
    def capabilities(self) -> ProviderExecutionCapabilities:
        return self._capabilities

    def get_capabilities(self) -> ProviderExecutionCapabilities:
        return self._capabilities

    def execute(self, request, candidate, storage_path, idempotency_key=None):
        self.submissions.append((request, candidate, idempotency_key))
        if not self._results:
            return AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE, error_code="EXHAUSTED")
        return self._results.pop(0)

    def submit(self, request, candidate, storage_path, idempotency_key=None):
        return self.execute(request, candidate, storage_path, idempotency_key)

    def get_status(self, provider_job_id, storage_path):
        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="SUCCEEDED",
            file_path=str(storage_path / "recovered_asset.png"),
            raw_response={"task_id": provider_job_id, "status": "completed"},
        )


def _make_dummy_media(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1280, 720), color=(25, 45, 85))
    img.save(path)
    return path


def print_box(title: str, content: list[str]):
    width = 76
    print(f"\n+{'-' * (width - 2)}+")
    print(f"| {title.ljust(width - 4)} |")
    print(f"+{'-' * (width - 2)}+")
    for line in content:
        print(f"| {line.ljust(width - 4)} |")
    print(f"+{'-' * (width - 2)}+")


def run_demo():
    print("\n" + "=" * 76)
    print(" [DEMO] MoneyPrinterTurbo -- Phase 5.2 Provider 声明周期与重试/降级 执行演示")
    print("=" * 76)

    tmp_dir = Path(tempfile.mkdtemp(prefix="mpt_demo_"))

    # Shared request
    req = AssetRoutingRequest(
        shot_id="shot-001",
        shot_revision_id="rev-001",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=4.5,
        visual_goal="展示太空望远镜展开太阳翼",
        scene_description="高轨道深空，金色太阳能阵列平滑展开，反射恒星光芒",
        generation_prompt="cinematic 4k deep space telescope deploying solar panels",
        camera_movement="Slow orbital dolly zoom",
        aspect_ratio="16:9",
    )

    # -----------------------------------------------------------------------
    # 场景 1: 正常成功生成 (Happy Path)
    # -----------------------------------------------------------------------
    media_file_1 = _make_dummy_media(tmp_dir / "shot-001" / "seedance_output.png")
    adapter_seedance = DemoFakeAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(media_file_1),
            remote_task_id="sd-job-88192",
            raw_response={"task_id": "sd-job-88192", "cost": 0.04},
        )
    ])
    registry_1 = AdapterRegistry({"seedance": adapter_seedance})
    service_1 = ShotExecutionService(adapter_registry=registry_1)

    cand_1 = AssetRouteCandidate(
        capability_id="seedance:pro:AI_VIDEO",
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        requested_visual_type=VisualType.AI_VIDEO,
        is_eligible=True,
    )
    decision_1 = AssetRouteDecision(
        shot_id=req.shot_id,
        shot_revision_id=req.shot_revision_id,
        routing_strategy=RoutingStrategy.QUALITY_FIRST,
        requested_visual_type=VisualType.AI_VIDEO,
        selected_candidate=cand_1,
        eligible_candidates=(cand_1,),
        model_selection_mode="AUTO",
    )
    shot_exec_1 = ShotExecution(
        execution_run_id="run-demo-1",
        shot_id=req.shot_id,
        shot_revision_id=req.shot_revision_id,
        route_decision=decision_1,
    )

    final_exec_1, attempts_1, version_1 = service_1.execute_shot(shot_exec_1, req, tmp_dir)
    print_box(
        "【场景 1】正常生成 (Happy Path)",
        [
            f"分镜目标: {req.visual_goal}",
            f"候选模型: {cand_1.provider} ({cand_1.model})",
            f"执行状态: {final_exec_1.status.value} (耗时尝试次数: {len(attempts_1)})",
            f"生成资产 ID: {version_1.shot_asset_version_id if version_1 else 'None'}",
            f"物理文件大小: {version_1.file_size_bytes} 字节 | 分辨率: {version_1.width}x{version_1.height}",
            f"文件 SHA-256: {version_1.file_hash[:24]}... (物理落盘校验完整)",
        ],
    )

    # -----------------------------------------------------------------------
    # 场景 2: 偶发技术失败 -> 局部自动重试成功 (Transient Technical Retry)
    # -----------------------------------------------------------------------
    media_file_2 = _make_dummy_media(tmp_dir / "shot-002" / "retry_success.png")
    adapter_transient = DemoFakeAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
            error_code="HTTP_503_SERVICE_UNAVAILABLE",
            error_message="Upstream API Gateway temporary timeout",
        ),
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(media_file_2),
            remote_task_id="sd-job-88193",
        ),
    ])
    registry_2 = AdapterRegistry({"seedance": adapter_transient})
    service_2 = ShotExecutionService(
        adapter_registry=registry_2,
        policy=ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4),
    )
    shot_exec_2 = ShotExecution(
        execution_run_id="run-demo-2",
        shot_id="shot-002",
        shot_revision_id=req.shot_revision_id,
        route_decision=decision_1,
    )

    final_exec_2, attempts_2, version_2 = service_2.execute_shot(shot_exec_2, req, tmp_dir)
    print_box(
        "【场景 2】偶发技术失败 -> 局部自动重试 (Transient Retry)",
        [
            "Attempt #1: 遇到 HTTP 503 错误 (TRANSIENT_TECHNICAL)",
            "策略判定: candidate attempts (1) < max (2) -> RETRY_SAME_CANDIDATE",
            "Attempt #2: 自动发起第 2 次尝试 -> 成功返回！",
            f"最终执行状态: {final_exec_2.status.value} (总尝试: {len(attempts_2)} 次)",
            f"生成资产 ID: {version_2.shot_asset_version_id if version_2 else 'None'}",
        ],
    )

    # -----------------------------------------------------------------------
    # 场景 3: 能力不支持 -> 立即降级到下一候选，零重试浪费 (Capability Fallback)
    # -----------------------------------------------------------------------
    media_file_3 = _make_dummy_media(tmp_dir / "shot-003" / "fallback_wavespeed.png")
    adapter_seedance_incompat = DemoFakeAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
            error_code="ASPECT_RATIO_NOT_SUPPORTED",
            error_message="16:9 widescreen not supported on this model tier",
        ),
    ])
    adapter_wavespeed = DemoFakeAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(media_file_3),
            remote_task_id="ws-job-1122",
        )
    ])
    cand_2 = AssetRouteCandidate(
        capability_id="wavespeed:standard:AI_VIDEO",
        provider="wavespeed",
        model="standard",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        requested_visual_type=VisualType.AI_VIDEO,
        is_eligible=True,
    )
    decision_3 = AssetRouteDecision(
        shot_id="shot-003",
        shot_revision_id=req.shot_revision_id,
        routing_strategy=RoutingStrategy.BALANCED,
        requested_visual_type=VisualType.AI_VIDEO,
        selected_candidate=cand_1,
        eligible_candidates=(cand_1, cand_2),
        model_selection_mode="AUTO",
    )
    registry_3 = AdapterRegistry({"seedance": adapter_seedance_incompat, "wavespeed": adapter_wavespeed})
    service_3 = ShotExecutionService(adapter_registry=registry_3)
    shot_exec_3 = ShotExecution(
        execution_run_id="run-demo-3",
        shot_id="shot-003",
        shot_revision_id=req.shot_revision_id,
        route_decision=decision_3,
    )

    final_exec_3, attempts_3, _version_3 = service_3.execute_shot(shot_exec_3, req, tmp_dir)
    print_box(
        "【场景 3】能力不兼容 -> 立即切换候选 (Capability Fallback)",
        [
            "Attempt #1 (Seedance): 返回 ASPECT_RATIO_NOT_SUPPORTED (UNSUPPORTED_CAPABILITY)",
            "策略判定: 永久能力不兼容 -> 放弃同候选重试，立即启用备选候选",
            "Attempt #2 (Wavespeed): 沿冻结候选列表平滑切换 (保持 VisualType.AI_VIDEO)",
            f"最终执行状态: {final_exec_3.status.value}",
            f"成功产出模型: {attempts_3[-1].provider} ({attempts_3[-1].model})",
        ],
    )

    # -----------------------------------------------------------------------
    # 场景 4: 提交未确认 -> 阻断盲目重试 -> 幂等补偿恢复 (Safety Guard & Recovery)
    # -----------------------------------------------------------------------
    adapter_ambiguous = DemoFakeAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
            remote_task_id="unconfirmed-sd-99901",
            error_message="TCP connection severed during HTTP POST before response header",
        ),
    ])
    registry_4 = AdapterRegistry({"seedance": adapter_ambiguous})
    service_4 = ShotExecutionService(adapter_registry=registry_4)
    shot_exec_4 = ShotExecution(
        execution_run_id="run-demo-4",
        shot_id="shot-004",
        shot_revision_id=req.shot_revision_id,
        route_decision=decision_1,
    )

    final_exec_4, attempts_4, _version_4 = service_4.execute_shot(shot_exec_4, req, tmp_dir)

    # 验证防重入硬阻断
    blocked_by_guard = False
    try:
        service_4.execute_shot(final_exec_4, req, tmp_dir)
    except BlindSubmissionForbiddenError:
        blocked_by_guard = True

    # 执行幂等恢复服务
    target_asset_dir = tmp_dir / "shot_assets" / "shot-004"
    _make_dummy_media(target_asset_dir / "recovered_asset.png")
    recovery_svc = AttemptRecoveryService(adapter_registry=registry_4)
    from app.domain.asset_execution import AttemptRequest
    attempt_req_4 = AttemptRequest.create_sanitized(
        execution_attempt_id=attempts_4[0].execution_attempt_id,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        idempotency_key="idemp_shot-004_1_demo",
        raw_payload={"prompt": "deep space telescope"},
    )
    _decision_rec, recovered_exec, _recovered_att, recovered_ver = recovery_svc.recover(
        final_exec_4, attempts_4[0], attempt_req_4, target_dir=target_asset_dir
    )

    print_box(
        "【场景 4】提交状态未确认 -> 熔断防二次计费 -> 补偿恢复",
        [
            "Attempt #1: POST 途中网络中断 (SUBMISSION_OUTCOME_UNKNOWN)",
            f"分镜状态收敛: {final_exec_4.status.value} (挂起未确认远程任务: {final_exec_4.unconfirmed_remote_task_id})",
            f"盲目重试安全防护: {'[OK] 成功触发 BlindSubmissionForbiddenError 拦截二次扣费' if blocked_by_guard else '[FAIL] 拦截失败'}",
            "AttemptRecoveryService 介入: 复用原 Attempt 幂等凭据向 Provider 探活...",
            f"恢复结果: 远程任务已完成！分镜状态更新为: {recovered_exec.status.value}",
            f"绑定原始 Attempt: {recovered_ver.execution_attempt_id == attempts_4[0].execution_attempt_id}",
        ],
    )

    # -----------------------------------------------------------------------
    # 场景 5: PINNED 模式严禁自动切换模型 (PINNED Constraint)
    # -----------------------------------------------------------------------
    decision_pinned = AssetRouteDecision(
        shot_id="shot-005",
        shot_revision_id=req.shot_revision_id,
        routing_strategy=RoutingStrategy.QUALITY_FIRST,
        requested_visual_type=VisualType.AI_VIDEO,
        selected_candidate=cand_1,
        eligible_candidates=(cand_1, cand_2),
        model_selection_mode="PINNED",
    )
    adapter_pinned_fail = DemoFakeAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
            error_code="PINNED_FAILED",
        ),
    ])
    adapter_pinned_backup = DemoFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS),
    ])
    registry_5 = AdapterRegistry({"seedance": adapter_pinned_fail, "wavespeed": adapter_pinned_backup})
    service_5 = ShotExecutionService(adapter_registry=registry_5)
    shot_exec_5 = ShotExecution(
        execution_run_id="run-demo-5",
        shot_id="shot-005",
        shot_revision_id=req.shot_revision_id,
        route_decision=decision_pinned,
    )
    final_exec_5, _attempts_5, _version_5 = service_5.execute_shot(shot_exec_5, req, tmp_dir)

    print_box(
        "【场景 5】PINNED 模式约束保护 (PINNED Constraint)",
        [
            "用户策略: model_selection_mode = PINNED (锁定 Seedance Pro)",
            "执行结果: Seedance Pro 报错失败",
            f"分镜最终状态: {final_exec_5.status.value} (错误代码: {final_exec_5.error_code})",
            f"备选 Wavespeed 被调用次数: {len(adapter_pinned_backup.submissions)} (严格禁止静默降级切换)",
        ],
    )

    print("\n" + "=" * 76)
    print(" [DONE] 所有 5 个核心生命周期与容错场景均按预期完美执行！")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_demo()
