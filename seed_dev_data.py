"""
Seed realistic demonstration data into storage/dev.db
for manual human visual inspection of Storyboard Workbench and Asset Execution.
"""

import sys
from pathlib import Path

from PIL import Image

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.domain.asset_execution import (
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ExecutionStatus,
    ShotAssetVersion,
    ShotExecution,
    ShotExecutionStatus,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    AssetRoutingRequest,
    GenerationMode,
    ModelSelectionMode,
    RoutingStrategy,
    ShotRoutePlanEntry,
    ShotRoutePlanStatus,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_approval import StoryboardApprovalRecord
from app.persistence.database_lifecycle import run_database_migrations
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardApprovalRepository,
    StoryboardRepository,
)
from app.persistence.session import get_session
from app.services.asset_probe_service import probe_media_file


def seed():
    run_database_migrations()

    with get_session() as session:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        sb_repo = StoryboardRepository(session)
        appr_repo = StoryboardApprovalRepository(session)
        route_repo = AssetRoutePlanRepository(session)
        exec_repo = ExecutionRepository(session)

        # 1. Beats
        b1 = ContentBeat(beat_id="b-001", beat_lineage_id="lin-b-001", beat_type=BeatType.HOOK, order=1, intent="火箭升空发射震撼开场", target_duration=4.5, importance=0.9)
        b2 = ContentBeat(beat_id="b-002", beat_lineage_id="lin-b-002", beat_type=BeatType.KNOWLEDGE, order=2, intent="拉格朗日L2轨道与太阳翼镜面展开", target_duration=5.0, importance=1.0)
        b3 = ContentBeat(beat_id="b-003", beat_lineage_id="lin-b-003", beat_type=BeatType.SUMMARY, order=3, intent="红外光谱洞穿星际尘埃揭示早期宇宙", target_duration=4.0, importance=0.8)

        plan = ContentPlanRevision(
            content_plan_revision_id="cpr-jwst-001",
            revision_number=1,
            topic="詹姆斯·韦伯太空望远镜：刺破黑暗的宇宙之眼",
            overall_target_duration=13.5,
            beats=(b1, b2, b3),
        )
        plan_repo.add_revision(plan)

        # 2. Shots
        s1 = Shot(shot_id="shot-jwst-01", beat_lineage_id="lin-b-001", local_order=1)
        s2 = Shot(shot_id="shot-jwst-02", beat_lineage_id="lin-b-002", local_order=1)
        s3 = Shot(shot_id="shot-jwst-03", beat_lineage_id="lin-b-003", local_order=1)
        for s in (s1, s2, s3):
            shot_repo.add_shot(s)

        # 3. Shot Revisions
        sr1 = ShotRevision(
            shot_revision_id="sr-jwst-01",
            shot_id="shot-jwst-01",
            revision_number=1,
            beat_lineage_id="lin-b-001",
            created_from_beat_instance_id="b-001",
            narration="当阿丽亚娜5号火箭划破长空，人类历史上最昂贵的空间观测站奔向深空。",
            target_duration=4.5,
            visual_goal="展示重型运载火箭尾焰喷涌，穿透云霄直冲近地轨道的磅礴气势",
            visual_type=VisualType.AI_VIDEO,
            scene_description="火箭底部炽热橘红色尾焰，云海在脚下翻腾，阳光在整流罩上反光",
            generation_prompt="cinematic 4k slow motion heavy rocket launch penetrating dense clouds into orbit",
            camera_movement="Low angle ascending tracking shot",
        )
        sr2 = ShotRevision(
            shot_revision_id="sr-jwst-02",
            shot_id="shot-jwst-02",
            revision_number=1,
            beat_lineage_id="lin-b-002",
            created_from_beat_instance_id="b-002",
            narration="距离地球150万公里的拉格朗日L2点，18面镀金铍镜如蜂巢般严丝合缝展开。",
            target_duration=5.0,
            visual_goal="展示蜂巢状金色主镜与五层聚酰亚胺遮阳薄膜在失重环境下缓缓张开",
            visual_type=VisualType.AI_VIDEO,
            scene_description="深邃漆黑宇宙背景下，巨大的金色六边形镜面精密折射出耀眼金光",
            generation_prompt="ultra detailed 8k James Webb space telescope unfolding gold mirrors in deep space",
            camera_movement="Smooth circular orbital dolly around the primary mirror",
        )
        sr3 = ShotRevision(
            shot_revision_id="sr-jwst-03",
            shot_id="shot-jwst-03",
            revision_number=1,
            beat_lineage_id="lin-b-003",
            created_from_beat_instance_id="b-003",
            narration="红外线穿透了百亿光年的星际尘埃，宇宙大爆炸后第一缕星光在此刻被清晰捕获。",
            target_duration=4.0,
            visual_goal="展现红外星云内部气体发光细节，遥远星系如同宝石般璀璨呈现",
            visual_type=VisualType.AI_VIDEO,
            scene_description="色彩斑斓的创生之柱深空红外全景，无数初生恒星闪耀",
            generation_prompt="astrophotography deep field infrared galaxies glowing through nebula dust",
            camera_movement="Slow inward zoom towards distant spiral galaxy",
        )
        for sr in (sr1, sr2, sr3):
            shot_repo.add_revision(sr)

        # 4. Draft and Approved Storyboard Snapshots
        draft_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-jwst-draft-01",
            content_plan_revision_id="cpr-jwst-001",
            shot_revision_ids=("sr-jwst-01", "sr-jwst-02", "sr-jwst-03"),
            snapshot_state=StoryboardSnapshotState.DRAFT,
        )
        sb_repo.add_snapshot(draft_snap)

        approved_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-jwst-approved",
            content_plan_revision_id="cpr-jwst-001",
            shot_revision_ids=("sr-jwst-01", "sr-jwst-02", "sr-jwst-03"),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        sb_repo.add_snapshot(approved_snap)

        approval_rec = StoryboardApprovalRecord(
            source_draft_snapshot_id="snap-jwst-draft-01",
            approved_storyboard_snapshot_id="snap-jwst-approved",
            content_plan_revision_id="cpr-jwst-001",
            exact_shot_revision_ids=("sr-jwst-01", "sr-jwst-02", "sr-jwst-03"),
            approved_by="admin_reviewer",
            user_note="分镜镜头叙事与视觉目标契合度极高，节奏把控优异，人工审核准予投产。",
        )
        appr_repo.add_approval_record(approval_rec)

        # 5. Route Plan
        cands = (
            AssetRouteCandidate(capability_id="seedance:pro:AI_VIDEO", provider="seedance", model="pro", generation_mode=GenerationMode.TEXT_TO_VIDEO, requested_visual_type=VisualType.AI_VIDEO, is_eligible=True),
            AssetRouteCandidate(capability_id="wavespeed:standard:AI_VIDEO", provider="wavespeed", model="standard", generation_mode=GenerationMode.TEXT_TO_VIDEO, requested_visual_type=VisualType.AI_VIDEO, is_eligible=True),
        )
        entries = []
        for s, sr, b in [(s1, sr1, b1), (s2, sr2, b2), (s3, sr3, b3)]:
            routing_req = AssetRoutingRequest(
                shot_id=s.shot_id,
                shot_revision_id=sr.shot_revision_id,
                requested_visual_type=VisualType.AI_VIDEO,
                target_duration=sr.target_duration,
                visual_goal=sr.visual_goal,
                scene_description=sr.scene_description,
                generation_prompt=sr.generation_prompt,
                camera_movement=sr.camera_movement,
            )
            dec = AssetRouteDecision(
                shot_id=s.shot_id,
                shot_revision_id=sr.shot_revision_id,
                routing_strategy=RoutingStrategy.QUALITY_FIRST,
                requested_visual_type=VisualType.AI_VIDEO,
                selected_candidate=cands[0],
                eligible_candidates=cands,
                model_selection_mode="AUTO",
            )
            e = ShotRoutePlanEntry(
                shot_id=s.shot_id,
                shot_revision_id=sr.shot_revision_id,
                beat_lineage_id=b.beat_lineage_id,
                requested_visual_type=VisualType.AI_VIDEO,
                asset_routing_request=routing_req,
                route_decision=dec,
                route_status=ShotRoutePlanStatus.ROUTED,
            )
            entries.append(e)

        route_plan = AssetRoutePlan(
            asset_route_plan_id="arp-jwst-001",
            storyboard_snapshot_id=approved_snap.storyboard_snapshot_id,
            content_plan_revision_id=plan.content_plan_revision_id,
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            routing_policy_version="v1.0",
            model_selection_mode=ModelSelectionMode.AUTO,
            shot_routes=tuple(entries),
            status=AssetRoutePlanStatus.READY,
            total_shots=3,
            routed_shots=3,
            blocked_shots=0,
        )
        route_repo.add_route_plan(route_plan)

        # 6. Execution Run & Physical Files
        run = ExecutionRun(
            execution_run_id="run-jwst-live",
            asset_route_plan_id=route_plan.asset_route_plan_id,
            storyboard_snapshot_id=approved_snap.storyboard_snapshot_id,
            total_shots=3,
            succeeded_shots=3,
            status=ExecutionStatus.SUCCEEDED,
        )
        exec_repo.add_execution_run(run)

        storage_dir = Path("storage/shot_assets").resolve()
        for idx, (s, sr, color) in enumerate([(s1, sr1, (20, 30, 80)), (s2, sr2, (180, 140, 20)), (s3, sr3, (80, 20, 70))], 1):
            shot_dir = storage_dir / s.shot_id
            shot_dir.mkdir(parents=True, exist_ok=True)
            img_path = shot_dir / f"{sr.shot_revision_id}_visual.png"
            img = Image.new("RGB", (1280, 720), color=color)
            img.save(img_path)

            probe = probe_media_file(img_path)
            att = ExecutionAttempt(
                execution_attempt_id=f"att-jwst-{idx}",
                execution_run_id=run.execution_run_id,
                shot_id=s.shot_id,
                shot_revision_id=sr.shot_revision_id,
                attempt_number=1,
                provider="seedance",
                model="pro",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                status=ExecutionAttemptStatus.SUCCEEDED,
            )
            exec_repo.add_execution_attempt(att)

            asset_ver = ShotAssetVersion(
                shot_asset_version_id=f"sav-jwst-{idx}",
                shot_id=s.shot_id,
                shot_revision_id=sr.shot_revision_id,
                execution_attempt_id=att.execution_attempt_id,
                file_path=str(img_path),
                file_hash=probe.file_hash,
                file_size_bytes=probe.file_size_bytes,
                media_type=probe.media_type,
                mime_type=probe.mime_type,
                width=probe.width,
                height=probe.height,
                duration_seconds=probe.duration_seconds,
                provider="seedance",
                model="pro",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
            )
            exec_repo.add_shot_asset_version(asset_ver)

            shot_exec = ShotExecution(
                shot_execution_id=f"se-jwst-{idx}",
                execution_run_id=run.execution_run_id,
                shot_id=s.shot_id,
                shot_revision_id=sr.shot_revision_id,
                route_decision=entries[idx-1].route_decision,
                status=ShotExecutionStatus.SUCCEEDED,
                produced_asset_version_id=asset_ver.shot_asset_version_id,
                attempt_ids=[att.execution_attempt_id],
            )
            exec_repo.add_shot_execution(shot_exec)

        session.commit()
        print("SUCCESS: Seed data populated into storage/dev.db successfully!")


if __name__ == "__main__":
    seed()
