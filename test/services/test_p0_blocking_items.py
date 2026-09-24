from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.services.material  # noqa: F401
from app.domain.asset_execution import (
    AdapterExecutionResult,
    ExecutionAttemptStatus,
    ExecutionRetryPolicy,
    ProviderOutcomeType,
    ShotExecution,
    ShotExecutionStatus,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutingRequest,
    GenerationMode,
    RoutingStrategy,
    StaticCapabilityMetadata,
    TierLevel,
    create_routing_request_from_shot_revision,
)
from app.domain.enums import VisualType
from app.domain.shot import ShotRevision
from app.models.schema import VideoAspect
from app.persistence.database_lifecycle import (
    init_database_on_startup,
    wait_for_database,
)
from app.persistence.models import Base
from app.persistence.repositories import ExecutionRepository
from app.services.asset_adapters.local_asset_adapter import LocalAssetAdapter
from app.services.asset_adapters.openai_image_adapter import OpenAIImageAdapter
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_adapters.seedance_adapter import SeedanceAdapter
from app.services.asset_adapters.stock_adapter import StockSearchAdapter
from app.services.asset_adapters.wavespeed_adapter import WaveSpeedAdapter
from app.services.asset_capability_registry import CoverrStockAdapter
from app.services.asset_probe_service import (
    InvalidAssetFileError,
)
from app.services.attempt_recovery_service import AttemptRecoveryService
from app.services.shot_execution_service import ShotExecutionService


def _create_sample_shot_revision(shot_id: str = "shot_test_01") -> ShotRevision:
    return ShotRevision(
        shot_id=shot_id,
        revision_number=1,
        beat_lineage_id="beat_lineage_01",
        created_from_beat_instance_id="beat_inst_01",
        narration="Test narration",
        visual_type=VisualType.AI_VIDEO,
        target_duration=5.0,
        visual_goal="Explain photosynthesis",
        scene_description="Green leaves in sunlight",
        generation_prompt="cinematic green leaves",
        camera_movement="slow zoom",
    )


def _create_sample_candidate(
    capability_id: str,
    provider: str,
    model: str,
    generation_mode: GenerationMode,
    requested_visual_type: VisualType = VisualType.AI_VIDEO,
) -> AssetRouteCandidate:
    return AssetRouteCandidate(
        capability_id=capability_id,
        provider=provider,
        model=model,
        generation_mode=generation_mode,
        requested_visual_type=requested_visual_type,
        is_eligible=True,
        rejection_reasons=(),
        static_metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.MEDIUM,
            cost_tier=TierLevel.MEDIUM,
            latency_tier=TierLevel.MEDIUM,
        ),
    )


def _create_sample_routing_request(
    shot_id: str = "shot_test_01",
    aspect_ratio: str = "16:9",
    visual_type: VisualType = VisualType.AI_VIDEO,
) -> AssetRoutingRequest:
    return AssetRoutingRequest(
        shot_id=shot_id,
        shot_revision_id=f"rev_{shot_id}_1",
        requested_visual_type=visual_type,
        target_duration=5.0,
        visual_goal="Explain photosynthesis",
        scene_description="Green leaves in sunlight",
        generation_prompt="cinematic green leaves",
        camera_movement="slow zoom",
        aspect_ratio=aspect_ratio,
    )


def _create_sample_route_decision(
    candidate: AssetRouteCandidate,
) -> AssetRouteDecision:
    return AssetRouteDecision(
        shot_id="shot_test_01",
        shot_revision_id="rev_shot_test_01_1",
        requested_visual_type=candidate.requested_visual_type,
        routing_strategy=RoutingStrategy.BALANCED,
        selected_candidate=candidate,
        eligible_candidates=(candidate,),
        rejected_candidates=(),
    )


def _create_dummy_image_file(path: Path) -> Path:
    from PIL import Image

    img = Image.new("RGB", (640, 360), color="blue")
    img.save(path, format="PNG")
    return path


class TestP0BlockingItems:
    """Automated verification suite for all 24 P0 run-blocking items."""

    # -------------------------------------------------------------
    # 1-3. Dependencies
    # -------------------------------------------------------------
    def test_item_01_to_03_dependency_imports(self):
        import alembic
        import psycopg
        import sqlalchemy

        assert sqlalchemy.__version__ is not None
        assert alembic.__version__ is not None
        assert psycopg.__version__ is not None

    # -------------------------------------------------------------
    # 6-8. Database Startup Readiness and Lifecycle
    # -------------------------------------------------------------
    def test_item_06_to_08_database_readiness_and_lifecycle(self):
        engine = create_engine("sqlite:///:memory:")
        ready = wait_for_database(engine=engine, timeout=2.0)
        assert ready is True

        with patch("app.persistence.database_lifecycle.create_db_engine", return_value=engine), patch(
            "app.persistence.database_lifecycle.run_database_migrations"
        ) as mock_migrate:
            success = init_database_on_startup(timeout=2.0)
            assert success is True
            mock_migrate.assert_called_once()

    # -------------------------------------------------------------
    # 9. Domain Consistency: AssetRoutingRequest.aspect_ratio
    # -------------------------------------------------------------
    def test_item_09_aspect_ratio_and_target_aspect_ratio_coercion(self):
        # Case A: aspect_ratio explicitly provided
        req1 = AssetRoutingRequest(
            shot_id="s1",
            shot_revision_id="r1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=4.0,
            visual_goal="g1",
            scene_description="d1",
            generation_prompt="p1",
            camera_movement="c1",
            aspect_ratio="16:9",
        )
        assert req1.aspect_ratio == "16:9"
        assert req1.target_aspect_ratio == "16:9"

        # Case B: target_aspect_ratio passed to dict or model validator
        raw_dict = {
            "shot_id": "s2",
            "shot_revision_id": "r2",
            "requested_visual_type": VisualType.AI_VIDEO,
            "target_duration": 4.0,
            "visual_goal": "g2",
            "scene_description": "d2",
            "generation_prompt": "p2",
            "camera_movement": "c2",
            "target_aspect_ratio": "9:16",
        }
        req2 = AssetRoutingRequest.model_validate(raw_dict)
        assert req2.aspect_ratio == "9:16"
        assert req2.target_aspect_ratio == "9:16"

        # Case C: factory function creates cleanly with target_aspect_ratio
        rev = _create_sample_shot_revision("s3")
        req3 = create_routing_request_from_shot_revision(rev, target_aspect_ratio="9:16")
        assert req3.aspect_ratio == "9:16"
        assert req3.target_aspect_ratio == "9:16"

    # -------------------------------------------------------------
    # 10. WaveSpeed Adapter function arguments
    # -------------------------------------------------------------
    def test_item_10_wavespeed_adapter_arguments(self, tmp_path):
        adapter = WaveSpeedAdapter()
        req = _create_sample_routing_request()
        cand = _create_sample_candidate(
            capability_id="wavespeed:test:TEXT_TO_VIDEO",
            provider="wavespeed",
            model="test-t2v",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )

        dummy_file = tmp_path / "ws_video.mp4"
        dummy_file.write_bytes(b"dummy")

        mock_material_item = MagicMock()
        mock_material_item.url = str(dummy_file)

        with patch(
            "app.services.material.generate_videos_wavespeed",
            return_value=[mock_material_item],
        ) as mock_gen:
            res = adapter.execute(req, cand, tmp_path)
            assert res.outcome_type == ProviderOutcomeType.SUCCESS
            assert res.file_path == str(dummy_file)
            mock_gen.assert_called_once_with(
                search_term=req.generation_prompt,
                minimum_duration=int(req.target_duration),
                video_aspect=VideoAspect.landscape,
            )

    # -------------------------------------------------------------
    # 11. OpenAI Image Adapter function arguments
    # -------------------------------------------------------------
    def test_item_11_openai_image_adapter_arguments(self, tmp_path):
        adapter = OpenAIImageAdapter()
        req = _create_sample_routing_request(aspect_ratio="9:16", visual_type=VisualType.AI_IMAGE)
        cand = _create_sample_candidate(
            capability_id="openai:dall-e-3:TEXT_TO_IMAGE",
            provider="openai",
            model="dall-e-3",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            requested_visual_type=VisualType.AI_IMAGE,
        )

        dummy_img = tmp_path / "img.png"
        dummy_img.write_bytes(b"dummy_image")

        mock_material_item = MagicMock()
        mock_material_item.url = str(dummy_img)

        with patch(
            "app.services.material.generate_images_openai",
            return_value=[mock_material_item],
        ) as mock_gen:
            res = adapter.execute(req, cand, tmp_path)
            assert res.outcome_type == ProviderOutcomeType.SUCCESS
            assert res.file_path == str(dummy_img)
            mock_gen.assert_called_once_with(
                search_term=req.generation_prompt,
                minimum_duration=int(req.target_duration),
                video_aspect=VideoAspect.portrait,
                save_dir=str(tmp_path),
            )

    # -------------------------------------------------------------
    # 12 & 13. Pexels and Pixabay Adapter function arguments
    # -------------------------------------------------------------
    def test_item_12_and_13_pexels_and_pixabay_adapter_arguments(self, tmp_path):
        adapter = StockSearchAdapter()
        req = _create_sample_routing_request()

        # Test Pixabay
        cand_pixabay = _create_sample_candidate(
            capability_id="pixabay:video:STOCK_SEARCH",
            provider="pixabay",
            model="pixabay-search",
            generation_mode=GenerationMode.STOCK_SEARCH,
            requested_visual_type=VisualType.STOCK_VIDEO,
        )
        dummy_pix = tmp_path / "pixabay.mp4"
        dummy_pix.write_bytes(b"dummy")
        mock_pix_item = MagicMock()
        mock_pix_item.url = str(dummy_pix)

        with patch(
            "app.services.material.search_videos_pixabay",
            return_value=[mock_pix_item],
        ) as mock_pix:
            res = adapter.execute(req, cand_pixabay, tmp_path)
            assert res.outcome_type == ProviderOutcomeType.SUCCESS
            mock_pix.assert_called_once_with(
                search_term=req.visual_goal,
                minimum_duration=int(req.target_duration),
                video_aspect=VideoAspect.landscape,
            )

        # Test Pexels
        cand_pexels = _create_sample_candidate(
            capability_id="pexels:video:STOCK_SEARCH",
            provider="pexels",
            model="pexels-search",
            generation_mode=GenerationMode.STOCK_SEARCH,
            requested_visual_type=VisualType.STOCK_VIDEO,
        )
        mock_pex_item = MagicMock()
        mock_pex_item.url = str(dummy_pix)

        with patch(
            "app.services.material.search_videos_pexels",
            return_value=[mock_pex_item],
        ) as mock_pex:
            res = adapter.execute(req, cand_pexels, tmp_path)
            assert res.outcome_type == ProviderOutcomeType.SUCCESS
            mock_pex.assert_called_once_with(
                search_term=req.visual_goal,
                minimum_duration=int(req.target_duration),
                video_aspect=VideoAspect.landscape,
            )

    # -------------------------------------------------------------
    # 14. Seedance Adapter function arguments
    # -------------------------------------------------------------
    def test_item_14_seedance_adapter_arguments(self, tmp_path):
        adapter = SeedanceAdapter()
        req = _create_sample_routing_request(aspect_ratio="9:16")
        cand = _create_sample_candidate(
            capability_id="volcengine_seedance:doubao:TEXT_TO_VIDEO",
            provider="volcengine_seedance",
            model="doubao-seedance-1-0-pro-250528",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )

        dummy_vid = tmp_path / "seedance.mp4"
        dummy_vid.write_bytes(b"dummy")
        mock_item = MagicMock()
        mock_item.url = str(dummy_vid)

        with patch(
            "app.services.volcengine_seedance.generate_videos",
            return_value=[mock_item],
        ) as mock_seedance_gen:
            res = adapter.execute(req, cand, tmp_path)
            assert res.outcome_type == ProviderOutcomeType.SUCCESS
            assert res.raw_response["provider"] == "volcengine_seedance"
            mock_seedance_gen.assert_called_once_with(
                search_term=req.generation_prompt,
                minimum_duration=int(req.target_duration),
                video_aspect=VideoAspect.portrait,
            )

    # -------------------------------------------------------------
    # 15. Unified volcengine_seedance and seedance provider names
    # -------------------------------------------------------------
    def test_item_15_provider_name_unification(self):
        reg = AdapterRegistry()
        ad_volc = reg.get_adapter("volcengine_seedance")
        ad_seed = reg.get_adapter("seedance")
        assert isinstance(ad_volc, SeedanceAdapter)
        assert isinstance(ad_seed, SeedanceAdapter)

    # -------------------------------------------------------------
    # 16. Coverr Adapter and Capability Table Check
    # -------------------------------------------------------------
    def test_item_16_coverr_adapter_and_capability(self, tmp_path):
        reg = AdapterRegistry()
        coverr_adapter = reg.get_adapter("coverr")
        assert isinstance(coverr_adapter, StockSearchAdapter)

        req = _create_sample_routing_request()
        cand = _create_sample_candidate(
            capability_id="coverr:video-search:STOCK_SEARCH",
            provider="coverr",
            model="coverr-video-search",
            generation_mode=GenerationMode.STOCK_SEARCH,
            requested_visual_type=VisualType.STOCK_VIDEO,
        )

        dummy_vid = tmp_path / "coverr.mp4"
        dummy_vid.write_bytes(b"dummy")
        mock_cov_item = MagicMock()
        mock_cov_item.url = str(dummy_vid)

        with patch(
            "app.services.material.search_videos_coverr",
            return_value=[mock_cov_item],
        ) as mock_cov:
            res = coverr_adapter.execute(req, cand, tmp_path)
            assert res.outcome_type == ProviderOutcomeType.SUCCESS
            mock_cov.assert_called_once_with(
                search_term=req.visual_goal,
                minimum_duration=int(req.target_duration),
                video_aspect=VideoAspect.landscape,
            )

        # Capability check: Coverr is enabled by default since adapter is supplied
        cap_prov = CoverrStockAdapter()
        caps = cap_prov.get_capabilities()
        assert caps[0].enabled is True

        # Can be disabled via configuration
        with patch("app.config.config.app.get", return_value=False):
            assert cap_prov.get_capabilities()[0].enabled is False

    # -------------------------------------------------------------
    # 17-19. Adapters for system_source_asset, system_user_asset, system_diagram
    # -------------------------------------------------------------
    def test_item_17_to_19_system_asset_adapters(self, tmp_path):
        reg = AdapterRegistry()
        src_adapter = reg.get_adapter("system_source_asset")
        user_adapter = reg.get_adapter("system_user_asset")
        diagram_adapter = reg.get_adapter("system_diagram")

        assert isinstance(src_adapter, LocalAssetAdapter)
        assert isinstance(user_adapter, LocalAssetAdapter)
        assert isinstance(diagram_adapter, LocalAssetAdapter)

        # 1. Source Asset Execution with valid file ref
        real_file = tmp_path / "paper_figure.png"
        real_file.write_bytes(b"fake_figure_bytes")

        cand_src = _create_sample_candidate(
            capability_id="system:source_asset_extractor:SOURCE_ASSET_USE",
            provider="system_source_asset",
            model="knowledge_figure_extractor",
            generation_mode=GenerationMode.SOURCE_ASSET_USE,
            requested_visual_type=VisualType.SOURCE_ASSET,
        )
        req_with_ref = AssetRoutingRequest(
            shot_id="shot_src_1",
            shot_revision_id="rev_src_1",
            requested_visual_type=VisualType.SOURCE_ASSET,
            target_duration=5.0,
            visual_goal="Show figure",
            scene_description="Figure from paper",
            generation_prompt="",
            camera_movement="",
            source_asset_refs=(str(real_file),),
        )
        dest_dir = tmp_path / "dest"
        res_src = src_adapter.execute(req_with_ref, cand_src, dest_dir)
        assert res_src.outcome_type == ProviderOutcomeType.SUCCESS
        assert Path(res_src.file_path).is_file()

        # 2. Source Asset Execution without valid file ref -> CAPABILITY_INCOMPATIBLE
        req_bad = AssetRoutingRequest(
            shot_id="shot_src_2",
            shot_revision_id="rev_src_2",
            requested_visual_type=VisualType.SOURCE_ASSET,
            target_duration=5.0,
            visual_goal="Show figure",
            scene_description="Figure from paper",
            generation_prompt="",
            camera_movement="",
            source_asset_refs=("non_existent_file.png",),
        )
        res_bad = src_adapter.execute(req_bad, cand_src, dest_dir)
        assert res_bad.outcome_type == ProviderOutcomeType.CAPABILITY_INCOMPATIBLE

        # 3. Diagram execution -> local PNG
        cand_diag = _create_sample_candidate(
            capability_id="system:diagram_renderer:DIAGRAM_RENDER",
            provider="system_diagram",
            model="diagram_placeholder",
            generation_mode=GenerationMode.DIAGRAM_RENDER,
            requested_visual_type=VisualType.DIAGRAM,
        )
        res_diag = diagram_adapter.execute(req_with_ref, cand_diag, dest_dir)
        assert res_diag.outcome_type == ProviderOutcomeType.SUCCESS
        assert Path(res_diag.file_path).suffix == ".png"

    def test_system_diagram_adapter_renders_probeable_png(self, tmp_path):
        from PIL import Image

        adapter = AdapterRegistry().get_adapter("system_diagram")
        candidate = _create_sample_candidate(
            capability_id="system:diagram_renderer:DIAGRAM_RENDER",
            provider="system_diagram",
            model="diagram_card_v1",
            generation_mode=GenerationMode.DIAGRAM_RENDER,
            requested_visual_type=VisualType.DIAGRAM,
        )
        request = AssetRoutingRequest(
            shot_id="shot-diagram-1",
            shot_revision_id="rev-diagram-1",
            requested_visual_type=VisualType.DIAGRAM,
            target_duration=4.0,
            visual_goal="展示 Ollama 与 Dify 如何构成本地知识库",
            scene_description="Ollama 提供本地模型，Dify 编排知识流程，最终形成知识库",
            generation_prompt="Ollama -> Dify -> 本地知识库",
            camera_movement="static",
            aspect_ratio="16:9",
        )

        result = adapter.execute(request, candidate, tmp_path / "diagram")

        assert result.outcome_type == ProviderOutcomeType.SUCCESS
        output_path = Path(result.file_path)
        assert output_path.suffix == ".png"
        with Image.open(output_path) as image:
            assert image.size == (1920, 1080)
            assert image.format == "PNG"

    def test_system_knowledge_card_adapter_renders_probeable_png(self, tmp_path):
        from PIL import Image

        adapter = AdapterRegistry().get_adapter("system_knowledge_card")
        assert isinstance(adapter, LocalAssetAdapter)

        candidate = _create_sample_candidate(
            capability_id="system:knowledge_card:TEXT_TO_IMAGE",
            provider="system_knowledge_card",
            model="knowledge_card_v1",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            requested_visual_type=VisualType.AI_IMAGE,
        )

        aspect_expected_sizes = {
            "16:9": (1920, 1080),
            "9:16": (1080, 1920),
            "1:1": (1080, 1080),
        }

        for ar, expected_size in aspect_expected_sizes.items():
            request = AssetRoutingRequest(
                shot_id=f"shot-kc-{ar.replace(':', '_')}",
                shot_revision_id=f"rev-kc-{ar.replace(':', '_')}",
                requested_visual_type=VisualType.AI_IMAGE,
                target_duration=4.0,
                visual_goal="知识图卡展示核心概念",
                scene_description="概念A与概念B的关系对比，关键特性展示",
                generation_prompt="A -> B -> 核心结论",
                camera_movement="static",
                aspect_ratio=ar,
            )

            result = adapter.execute(request, candidate, tmp_path / f"kc_{ar.replace(':', '_')}")

            assert result.outcome_type == ProviderOutcomeType.SUCCESS
            assert result.file_path is not None
            output_path = Path(result.file_path)
            assert output_path.is_file()
            assert output_path.stat().st_size > 0
            assert output_path.suffix == ".png"
            with Image.open(output_path) as image:
                assert image.size == expected_size
                assert image.format == "PNG"

            assert result.raw_response.get("provider") == "system_knowledge_card"
            assert result.raw_response.get("renderer") == "knowledge_card_v1"

        # Verify error handling returns DEFINITIVE_TECHNICAL_FAILURE
        with patch("app.services.asset_adapters.local_asset_adapter.render_diagram_card", side_effect=RuntimeError("card render error")):
            fail_result = adapter.execute(request, candidate, tmp_path / "kc_fail")
            assert fail_result.outcome_type == ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE
            assert fail_result.error_code in ("KNOWLEDGE_CARD_RENDER_FAILED", "DIAGRAM_RENDER_FAILED")

    # -------------------------------------------------------------
    # 20 & 21. Truly Persist AttemptRequest & Re-read Idempotency Key
    # -------------------------------------------------------------
    def test_item_20_and_21_attempt_request_persistence_and_recovery(self, tmp_path):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)

        cand = _create_sample_candidate(
            capability_id="test:mock:TEXT_TO_VIDEO",
            provider="mock_prov",
            model="mock_model",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )
        decision = _create_sample_route_decision(cand)
        shot_exec = ShotExecution(
            execution_run_id="run_p0_test",
            shot_id="shot_p0_persist",
            shot_revision_id="rev_p0_persist_1",
            route_decision=decision,
            status=ShotExecutionStatus.PENDING,
        )
        req = _create_sample_routing_request(shot_id="shot_p0_persist")

        persisted_attempt_id = None
        saved_idempotency_key = None

        mock_adapter = MagicMock()

        def fake_submit(r, c, storage, idempotency_key=None):
            nonlocal persisted_attempt_id, saved_idempotency_key
            saved_idempotency_key = idempotency_key
            # Verify DB already contains the AttemptRequest at the moment of external submit!
            with session_factory() as session:
                # Find the persisted request by querying the single attempt in DB
                from sqlalchemy import select

                from app.persistence.models import AttemptRequestORM
                orm_req = session.scalars(select(AttemptRequestORM)).first()
                assert orm_req is not None
                assert orm_req.idempotency_key == idempotency_key
                persisted_attempt_id = orm_req.execution_attempt_id

            # Return UNKNOWN outcome so we can test recovery!
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                remote_task_id="remote_job_12345",
                error_code="SUBMISSION_OUTCOME_UNKNOWN",
            )

        mock_adapter.submit.side_effect = fake_submit
        reg = AdapterRegistry(custom_adapters={"mock_prov": mock_adapter})

        service = ShotExecutionService(
            adapter_registry=reg,
            session_factory=session_factory,
        )

        final_exec, attempts, _asset_ver = service.execute_shot(
            shot_execution=shot_exec,
            request=req,
            storage_base_dir=tmp_path,
        )

        assert final_exec.status == ShotExecutionStatus.NEEDS_RECOVERY
        assert len(attempts) == 1
        assert attempts[0].status == ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN

        # Verify idempotency key can be safely re-read from DB on task recovery
        with session_factory() as session:
            repo = ExecutionRepository(session)
            re_read_req = repo.get_attempt_request(persisted_attempt_id)
            assert re_read_req is not None
            assert re_read_req.idempotency_key == saved_idempotency_key

        # Test AttemptRecoveryService using re-read AttemptRequest
        mock_adapter.get_capabilities.return_value = MagicMock(
            supports_async_status=True,
            supports_safe_resubmission_with_same_key=False,
        )
        dummy_local_img = tmp_path / "recovered.png"
        _create_dummy_image_file(dummy_local_img)
        mock_adapter.get_status.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(dummy_local_img),
        )

        recovery_svc = AttemptRecoveryService(adapter_registry=reg)
        rec_dec, _rec_exec, rec_att, rec_ver = recovery_svc.recover(
            shot_execution=final_exec,
            attempt=attempts[0],
            attempt_request=re_read_req,
            target_dir=tmp_path,
        )

        assert rec_dec.action.value == "RESOLVED_SUCCEEDED"
        assert rec_att.status == ExecutionAttemptStatus.SUCCEEDED
        assert rec_ver is not None

    # -------------------------------------------------------------
    # 22-24. Media URL Download, Probing, and Explicit Failure States
    # -------------------------------------------------------------
    def test_item_22_to_24_media_url_download_probe_and_error_handling(self, tmp_path):
        cand = _create_sample_candidate(
            capability_id="test:mock:TEXT_TO_VIDEO",
            provider="mock_media_prov",
            model="mock_model",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )
        decision = _create_sample_route_decision(cand)
        req = _create_sample_routing_request()

        # Case 1: URL returned -> download succeeds -> probe succeeds -> ShotAssetVersion produced
        local_png = tmp_path / "valid_image.png"
        _create_dummy_image_file(local_png)

        mock_adapter_1 = MagicMock()
        mock_adapter_1.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path="https://example.com/asset.png",
        )
        reg_1 = AdapterRegistry(custom_adapters={"mock_media_prov": mock_adapter_1})
        service_1 = ShotExecutionService(adapter_registry=reg_1)

        shot_exec_1 = ShotExecution(
            execution_run_id="run_1",
            shot_id="shot_dl_1",
            shot_revision_id="rev_dl_1",
            route_decision=decision,
            status=ShotExecutionStatus.PENDING,
        )

        with patch(
            "app.services.asset_probe_service.download_media_file",
            return_value=local_png,
        ) as mock_dl:
            final_exec, _attempts, asset_ver = service_1.execute_shot(
                shot_execution=shot_exec_1,
                request=req,
                storage_base_dir=tmp_path,
            )
            mock_dl.assert_called_once()
            assert final_exec.status == ShotExecutionStatus.SUCCEEDED
            assert asset_ver is not None
            assert asset_ver.width == 640
            assert asset_ver.height == 360

        # Case 2: URL returned -> download fails -> attempt marked FAILED (DOWNLOAD_FAILED)
        mock_adapter_2 = MagicMock()
        mock_adapter_2.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path="https://example.com/bad_download.mp4",
        )
        reg_2 = AdapterRegistry(custom_adapters={"mock_media_prov": mock_adapter_2})
        service_2 = ShotExecutionService(
            adapter_registry=reg_2,
            policy=ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1),
        )
        shot_exec_2 = ShotExecution(
            execution_run_id="run_2",
            shot_id="shot_dl_2",
            shot_revision_id="rev_dl_2",
            route_decision=decision,
            status=ShotExecutionStatus.PENDING,
        )

        with patch(
            "app.services.asset_probe_service.download_media_file",
            side_effect=InvalidAssetFileError("Remote media URL returned HTTP 404"),
        ):
            final_exec_2, attempts_2, asset_ver_2 = service_2.execute_shot(
                shot_execution=shot_exec_2,
                request=req,
                storage_base_dir=tmp_path,
            )
            assert final_exec_2.status in (ShotExecutionStatus.FAILED, ShotExecutionStatus.EXHAUSTED)
            assert attempts_2[0].status == ExecutionAttemptStatus.FAILED
            assert attempts_2[0].error_code == "DOWNLOAD_FAILED"
            assert asset_ver_2 is None

        # Case 3: Empty file (0 bytes) -> marked FAILED (EMPTY_MEDIA_FILE)
        empty_file = tmp_path / "empty.mp4"
        empty_file.write_bytes(b"")
        mock_adapter_3 = MagicMock()
        mock_adapter_3.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(empty_file),
        )
        reg_3 = AdapterRegistry(custom_adapters={"mock_media_prov": mock_adapter_3})
        service_3 = ShotExecutionService(
            adapter_registry=reg_3,
            policy=ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1),
        )
        shot_exec_3 = ShotExecution(
            execution_run_id="run_3",
            shot_id="shot_dl_3",
            shot_revision_id="rev_dl_3",
            route_decision=decision,
            status=ShotExecutionStatus.PENDING,
        )

        final_exec_3, attempts_3, _asset_ver_3 = service_3.execute_shot(
            shot_execution=shot_exec_3,
            request=req,
            storage_base_dir=tmp_path,
        )
        assert final_exec_3.status in (ShotExecutionStatus.FAILED, ShotExecutionStatus.EXHAUSTED)
        assert attempts_3[0].status == ExecutionAttemptStatus.FAILED
        assert attempts_3[0].error_code == "EMPTY_MEDIA_FILE"

        # Case 4: Format error / probe failure -> marked FAILED (INVALID_ASSET_FORMAT)
        corrupt_file = tmp_path / "corrupt.png"
        corrupt_file.write_bytes(b"not_an_image_or_video_corrupt_data")
        mock_adapter_4 = MagicMock()
        mock_adapter_4.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(corrupt_file),
        )
        reg_4 = AdapterRegistry(custom_adapters={"mock_media_prov": mock_adapter_4})
        service_4 = ShotExecutionService(
            adapter_registry=reg_4,
            policy=ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1),
        )
        shot_exec_4 = ShotExecution(
            execution_run_id="run_4",
            shot_id="shot_dl_4",
            shot_revision_id="rev_dl_4",
            route_decision=decision,
            status=ShotExecutionStatus.PENDING,
        )

        final_exec_4, attempts_4, _asset_ver_4 = service_4.execute_shot(
            shot_execution=shot_exec_4,
            request=req,
            storage_base_dir=tmp_path,
        )
        assert final_exec_4.status in (ShotExecutionStatus.FAILED, ShotExecutionStatus.EXHAUSTED)
        assert attempts_4[0].status == ExecutionAttemptStatus.FAILED
        assert attempts_4[0].error_code == "INVALID_ASSET_FORMAT"
