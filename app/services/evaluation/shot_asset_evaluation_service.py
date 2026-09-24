from __future__ import annotations

from sqlalchemy.orm import Session

from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationDimension,
    EvaluationMediaUnreadableError,
    EvaluationReasonCode,
    EvaluationSnapshot,
    EvaluationTarget,
    create_evaluation_snapshot,
)
from app.domain.evaluation_policy import (
    EvaluationPolicy,
    EvaluationPolicyDecision,
    EvaluationPolicyEngine,
)
from app.persistence.repositories import (
    EvaluationRepository,
    ExecutionRepository,
    ShotRepository,
)
from app.services.evaluation.composition_preview import (
    CompositionPreview,
    CompositionPreviewService,
    RenderContext,
)
from app.services.evaluation.dependency_fingerprint import (
    COMPOSITION_SUITABILITY_SEMANTICS_VERSION,
    KNOWLEDGE_ACCURACY_SEMANTICS_VERSION,
    SEMANTIC_ALIGNMENT_SEMANTICS_VERSION,
    VISUAL_QUALITY_SEMANTICS_VERSION,
    compute_composition_suitability_fingerprint,
    compute_knowledge_accuracy_fingerprint,
    compute_semantic_alignment_fingerprint,
    compute_visual_quality_fingerprint,
)
from app.services.evaluation.dimension_evaluators import (
    CompositionSuitabilityEvaluator,
    KnowledgeAccuracyEvaluator,
    SemanticAlignmentEvaluator,
    VisualQualityEvaluator,
)
from app.services.evaluation.media_preparation import (
    PreparedMediaObservation,
    prepare_media_observation,
)
from app.services.evaluation.mpt_multimodal_adapter import (
    MultimodalEvaluatorAdapter,
)


class ShotAssetEvaluationService:
    """
    Orchestrates Shot-level multimodal evaluation for an exact immutable EvaluationTarget.

    Responsibilities:
    1. Loads target, shot revision, and asset version.
    2. Bounded media preparation & frame sampling.
    3. Derived composition preview generation/caching.
    4. Deterministic dependency fingerprint calculation.
    5. Safe reuse of compatible historical SCORED results.
    6. Multimodal evaluation of missing dimensions outside DB transaction locks.
    7. EvaluationPolicyEngine execution.
    8. Immutable EvaluationSnapshot persistence.
    9. Strict adherence to architectural boundaries (Zero Phase 6.3 regeneration).
    """

    def __init__(self, session: Session):
        self._session = session
        self._eval_repo = EvaluationRepository(session)
        self._shot_repo = ShotRepository(session)
        self._exec_repo = ExecutionRepository(session)

    def evaluate_target(
        self,
        evaluation_target_id: str,
        policy: EvaluationPolicy | None = None,
        render_context: RenderContext | None = None,
        force_re_evaluation: bool = False,
        evaluator_adapter: MultimodalEvaluatorAdapter | None = None,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> EvaluationSnapshot:
        """
        Executes evaluation for an immutable EvaluationTarget.
        """
        active_policy = policy or EvaluationPolicy()
        adapter = evaluator_adapter or MultimodalEvaluatorAdapter()
        ctx = render_context or RenderContext()

        # 1. Load target context
        target = self._eval_repo.get_target(evaluation_target_id)
        if target is None:
            raise ValueError(f"EvaluationTarget '{evaluation_target_id}' not found")

        shot_rev = self._shot_repo.get_revision(target.shot_revision_id)
        if shot_rev is None:
            raise ValueError(f"ShotRevision '{target.shot_revision_id}' not found")

        asset_ver = self._exec_repo.get_shot_asset_version(target.shot_asset_version_id)
        if asset_ver is None:
            raise ValueError(f"ShotAssetVersion '{target.shot_asset_version_id}' not found")

        eval_ctx = None
        eval_ev = None
        if trace_writer is not None and trace_context is not None:
            from app.domain.trace import TraceEventType

            eval_ctx = trace_context.child_context(
                parent_event_id=trace_context.parent_event_id,
                evaluation_target_id=evaluation_target_id,
                shot_id=shot_rev.shot_id if hasattr(shot_rev, "shot_id") else None,
                shot_revision_id=target.shot_revision_id,
                shot_asset_version_id=target.shot_asset_version_id,
            )
            eval_ev = trace_writer.start_event(
                context=eval_ctx,
                event_type=TraceEventType.EVALUATION_STARTED,
                attributes={
                    "evaluation_target_id": evaluation_target_id,
                    "shot_id": shot_rev.shot_id if hasattr(shot_rev, "shot_id") else None,
                    "shot_revision_id": target.shot_revision_id,
                    "shot_asset_version_id": target.shot_asset_version_id,
                    "policy_version": active_policy.policy_version,
                    "evaluator_version": adapter.evaluator_version,
                },
            )

        # 2. Media preparation with error containment
        media_obs: PreparedMediaObservation | None = None
        media_unreadable_error: str | None = None

        try:
            media_obs = prepare_media_observation(
                asset_ver.file_path,
                media_type=asset_ver.media_type,
            )
        except EvaluationMediaUnreadableError as exc:
            media_unreadable_error = str(exc)

        # If media cannot be read or probed, all dimensions produce unreadable media ERROR
        if media_obs is None:
            return self._handle_unreadable_media_evaluation(
                target=target,
                error_message=media_unreadable_error or "Media unreadable",
                evaluator_version=adapter.evaluator_version,
                policy=active_policy,
            )

        # 3. Composition preview generation or retrieval
        comp_preview = self._resolve_composition_preview(
            shot_asset_version_id=target.shot_asset_version_id,
            source_image_path=media_obs.frame_paths[0],
            render_context=ctx,
        )

        # 4. Compute dependency fingerprints for all 4 dimensions
        fp_semantic = compute_semantic_alignment_fingerprint(
            asset_file_hash=media_obs.asset_file_hash,
            narration=shot_rev.narration,
            visual_goal=shot_rev.visual_goal,
            scene_description=shot_rev.scene_description,
            generation_prompt=shot_rev.generation_prompt,
            visual_type=shot_rev.visual_type.value if hasattr(shot_rev.visual_type, "value") else str(shot_rev.visual_type),
            evaluator_version=adapter.evaluator_version,
            semantics_version=SEMANTIC_ALIGNMENT_SEMANTICS_VERSION,
            sampling_policy_version=media_obs.sampling_policy_version,
        )

        fp_visual = compute_visual_quality_fingerprint(
            asset_file_hash=media_obs.asset_file_hash,
            evaluator_version=adapter.evaluator_version,
            semantics_version=VISUAL_QUALITY_SEMANTICS_VERSION,
            sampling_policy_version=media_obs.sampling_policy_version,
        )

        fp_knowledge = compute_knowledge_accuracy_fingerprint(
            asset_file_hash=media_obs.asset_file_hash,
            narration=shot_rev.narration,
            evidence_refs=target.evidence_refs_snapshot,
            evaluator_version=adapter.evaluator_version,
            semantics_version=KNOWLEDGE_ACCURACY_SEMANTICS_VERSION,
            sampling_policy_version=media_obs.sampling_policy_version,
        )

        fp_composition = compute_composition_suitability_fingerprint(
            preview_file_hash=comp_preview.file_hash,
            render_context_fingerprint=comp_preview.render_context_fingerprint,
            evaluator_version=adapter.evaluator_version,
            semantics_version=COMPOSITION_SUITABILITY_SEMANTICS_VERSION,
            preview_policy_version=comp_preview.preview_policy_version,
        )

        expected_fingerprints = {
            EvaluationDimension.SEMANTIC_ALIGNMENT: fp_semantic,
            EvaluationDimension.VISUAL_QUALITY: fp_visual,
            EvaluationDimension.KNOWLEDGE_ACCURACY: fp_knowledge,
            EvaluationDimension.COMPOSITION_SUITABILITY: fp_composition,
        }

        # 5. Check historical results for target and reuse compatible SCORED results
        historical_results = self._eval_repo.list_dimension_results_for_target(
            target.evaluation_target_id
        )

        reused_results: dict[EvaluationDimension, DimensionEvaluationResult] = {}
        if not force_re_evaluation:
            for res in reversed(historical_results):
                dim = res.dimension
                if (
                    dim not in reused_results
                    and res.status == DimensionEvaluationStatus.SCORED
                    and res.dependency_fingerprint == expected_fingerprints[dim]
                    and res.evaluator_version == adapter.evaluator_version
                ):
                    reused_results[dim] = res

        # 6. Evaluate un-reused dimensions outside DB transaction
        final_results: list[DimensionEvaluationResult] = []

        # Dimension 1: SEMANTIC_ALIGNMENT
        if EvaluationDimension.SEMANTIC_ALIGNMENT in reused_results:
            final_results.append(reused_results[EvaluationDimension.SEMANTIC_ALIGNMENT])
        else:
            res = SemanticAlignmentEvaluator.evaluate(
                evaluation_target_id=target.evaluation_target_id,
                media_observation=media_obs,
                narration=shot_rev.narration,
                visual_goal=shot_rev.visual_goal,
                scene_description=shot_rev.scene_description,
                generation_prompt=shot_rev.generation_prompt,
                visual_type=shot_rev.visual_type.value if hasattr(shot_rev.visual_type, "value") else str(shot_rev.visual_type),
                adapter=adapter,
            )
            saved_res = self._eval_repo.save_dimension_result(res)
            self._session.commit()
            final_results.append(saved_res)

        # Dimension 2: VISUAL_QUALITY
        if EvaluationDimension.VISUAL_QUALITY in reused_results:
            final_results.append(reused_results[EvaluationDimension.VISUAL_QUALITY])
        else:
            res = VisualQualityEvaluator.evaluate(
                evaluation_target_id=target.evaluation_target_id,
                media_observation=media_obs,
                adapter=adapter,
            )
            saved_res = self._eval_repo.save_dimension_result(res)
            self._session.commit()
            final_results.append(saved_res)

        # Dimension 3: KNOWLEDGE_ACCURACY
        if EvaluationDimension.KNOWLEDGE_ACCURACY in reused_results:
            final_results.append(reused_results[EvaluationDimension.KNOWLEDGE_ACCURACY])
        else:
            res = KnowledgeAccuracyEvaluator.evaluate(
                evaluation_target_id=target.evaluation_target_id,
                media_observation=media_obs,
                narration=shot_rev.narration,
                evidence_refs=target.evidence_refs_snapshot,
                adapter=adapter,
            )
            saved_res = self._eval_repo.save_dimension_result(res)
            self._session.commit()
            final_results.append(saved_res)

        # Dimension 4: COMPOSITION_SUITABILITY
        if EvaluationDimension.COMPOSITION_SUITABILITY in reused_results:
            final_results.append(reused_results[EvaluationDimension.COMPOSITION_SUITABILITY])
        else:
            res = CompositionSuitabilityEvaluator.evaluate(
                evaluation_target_id=target.evaluation_target_id,
                composition_preview=comp_preview,
                render_context=ctx,
                adapter=adapter,
            )
            saved_res = self._eval_repo.save_dimension_result(res)
            self._session.commit()
            final_results.append(saved_res)

        # 7. Execute deterministic Policy Engine
        policy_decision: EvaluationPolicyDecision = EvaluationPolicyEngine.evaluate(
            dimension_results=final_results,
            policy=active_policy,
        )

        # 8. Create and persist immutable EvaluationSnapshot
        snapshot = create_evaluation_snapshot(
            target=target,
            dimension_results=final_results,
            decision=policy_decision.decision,
            policy_version=active_policy.policy_version,
            evaluator_version=adapter.evaluator_version,
            overall_score=policy_decision.overall_score,
            summary_reason_codes=policy_decision.reason_codes,
        )

        saved_snapshot = self._eval_repo.save_snapshot(snapshot)
        self._session.commit()

        if trace_writer is not None and eval_ctx is not None:
            from app.domain.trace import (
                EvaluationCompletedTraceData,
                TraceEventStatus,
                TraceEventType,
            )

            for dim_res in final_results:
                trace_writer.record_event(
                    context=eval_ctx,
                    event_type=TraceEventType.DIMENSION_EVALUATION_COMPLETED,
                    status=TraceEventStatus.SUCCEEDED
                    if dim_res.status == DimensionEvaluationStatus.SCORED
                    else TraceEventStatus.FAILED,
                    attributes={
                        "evaluation_target_id": evaluation_target_id,
                        "dimension": dim_res.dimension.value
                        if hasattr(dim_res.dimension, "value")
                        else str(dim_res.dimension),
                        "status": dim_res.status.value
                        if hasattr(dim_res.status, "value")
                        else str(dim_res.status),
                        "score": dim_res.score,
                        "reason_codes": list(dim_res.reason_codes),
                        "dependency_fingerprint": dim_res.dependency_fingerprint,
                        "is_reused": dim_res.dimension in reused_results,
                    },
                    parent_event_id=eval_ev.trace_event_id if eval_ev else None,
                )

            eval_comp_data = EvaluationCompletedTraceData(
                evaluation_target_id=target.evaluation_target_id,
                evaluation_snapshot_id=saved_snapshot.evaluation_snapshot_id,
                shot_asset_version_id=target.shot_asset_version_id,
                evaluator_provider=getattr(adapter, "provider", None) or "multimodal_adapter",
                evaluator_model=getattr(adapter, "model", None) or adapter.evaluator_version,
                evaluation_policy_version=active_policy.policy_version,
                dimension_statuses={
                    (d.dimension.value if hasattr(d.dimension, "value") else str(d.dimension)): (
                        d.status.value if hasattr(d.status, "value") else str(d.status)
                    )
                    for d in final_results
                },
                dimension_scores={
                    (d.dimension.value if hasattr(d.dimension, "value") else str(d.dimension)): d.score
                    for d in final_results
                },
                decision=policy_decision.decision.value
                if hasattr(policy_decision.decision, "value")
                else str(policy_decision.decision),
                reused_dimension_count=len(reused_results),
                evaluator_call_count=len(final_results) - len(reused_results),
            )

            if eval_ev is not None:
                trace_writer.complete_event(
                    event=eval_ev,
                    status=TraceEventStatus.SUCCEEDED,
                    attributes_update=eval_comp_data,
                )

            trace_writer.record_event(
                context=eval_ctx.child_context(
                    parent_event_id=eval_ev.trace_event_id if eval_ev else None,
                    evaluation_snapshot_id=saved_snapshot.evaluation_snapshot_id,
                ),
                event_type=TraceEventType.EVALUATION_COMPLETED,
                status=TraceEventStatus.SUCCEEDED,
                attributes=eval_comp_data,
                parent_event_id=eval_ev.trace_event_id if eval_ev else None,
            )

        return saved_snapshot

    def _resolve_composition_preview(
        self,
        shot_asset_version_id: str,
        source_image_path: str,
        render_context: RenderContext,
    ) -> CompositionPreview:
        """Retrieves or generates derived composition preview and stores in repository."""
        ctx_fp = render_context.compute_fingerprint()
        existing = self._eval_repo.get_composition_preview_by_fingerprint(
            shot_asset_version_id=shot_asset_version_id,
            render_context_fingerprint=ctx_fp,
        )
        if existing is not None:
            return existing

        preview = CompositionPreviewService.generate_or_get_preview(
            shot_asset_version_id=shot_asset_version_id,
            source_image_path=source_image_path,
            render_context=render_context,
        )
        self._eval_repo.save_composition_preview(preview)
        self._session.commit()
        return preview

    def _handle_unreadable_media_evaluation(
        self,
        target: EvaluationTarget,
        error_message: str,
        evaluator_version: str,
        policy: EvaluationPolicy,
    ) -> EvaluationSnapshot:
        """
        Handles unreadable or corrupted media files:
        Marks all four dimensions as ERROR, resulting in INDETERMINATE snapshot decision.
        """
        dim_results: list[DimensionEvaluationResult] = []
        for dim in EvaluationDimension:
            res = DimensionEvaluationResult(
                evaluation_target_id=target.evaluation_target_id,
                dimension=dim,
                status=DimensionEvaluationStatus.ERROR,
                score=None,
                reason_codes=(EvaluationReasonCode.EVALUATION_MEDIA_UNREADABLE.value,),
                concise_summary=f"Media unreadable: {error_message[:150]}",
                evaluator_version=evaluator_version,
                error_code=EvaluationReasonCode.EVALUATION_MEDIA_UNREADABLE.value,
            )
            saved = self._eval_repo.save_dimension_result(res)
            dim_results.append(saved)

        self._session.commit()

        policy_decision = EvaluationPolicyEngine.evaluate(dim_results, policy=policy)

        snapshot = create_evaluation_snapshot(
            target=target,
            dimension_results=dim_results,
            decision=policy_decision.decision,
            policy_version=policy.policy_version,
            evaluator_version=evaluator_version,
            overall_score=None,
            summary_reason_codes=policy_decision.reason_codes,
        )

        saved_snap = self._eval_repo.save_snapshot(snapshot)
        self._session.commit()
        return saved_snap
