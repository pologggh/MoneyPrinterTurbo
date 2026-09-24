from __future__ import annotations

import shutil
from pathlib import Path

from app.domain.asset_execution import AdapterExecutionResult, ProviderOutcomeType
from app.domain.asset_router import AssetRouteCandidate, AssetRoutingRequest
from app.services.asset_adapters.base import AssetExecutionAdapter


class LocalAssetAdapter(AssetExecutionAdapter):
    """
    Adapter for resolving local system assets:
    - system_source_asset: original knowledge source figures/images
    - system_user_asset: user-supplied uploaded media
    - system_diagram: conceptual diagram renderer (disabled by default)
    """

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        target_dir.mkdir(parents=True, exist_ok=True)
        provider = candidate.provider.lower()

        if provider == "system_diagram":
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                error_code="DIAGRAM_ENGINE_UNAVAILABLE",
                error_message="Diagram rendering engine is not configured or enabled",
            )

        refs: tuple[str, ...] = ()
        if provider in ("system_source_asset", "source_asset"):
            refs = request.source_asset_refs
        elif provider in ("system_user_asset", "user_asset"):
            refs = request.user_asset_refs
        else:
            refs = request.user_asset_refs or request.source_asset_refs

        if not refs:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                error_code="NO_LOCAL_ASSET_REFERENCES",
                error_message=f"No asset references supplied for {candidate.provider}",
            )

        # Locate first valid file reference
        valid_path: Path | None = None
        for ref in refs:
            candidate_path = Path(ref)
            if candidate_path.is_file() and candidate_path.stat().st_size > 0:
                valid_path = candidate_path
                break

        if valid_path is None:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                error_code="LOCAL_ASSET_FILE_NOT_FOUND",
                error_message=f"Referenced files for {candidate.provider} do not exist or are empty: {refs}",
            )

        # Copy asset into target directory to ensure local persistence and isolation
        dest_filename = f"{candidate.provider}_{valid_path.name}"
        dest_path = target_dir / dest_filename
        try:
            if valid_path.resolve() != dest_path.resolve():
                shutil.copy2(valid_path, dest_path)
            else:
                dest_path = valid_path

            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                file_path=str(dest_path),
                raw_response={"provider": candidate.provider, "original_ref": str(valid_path)},
            )
        except Exception as exc:  # noqa: BLE001
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                error_code="LOCAL_ASSET_COPY_FAILED",
                error_message=f"Failed to copy local asset {valid_path} to {dest_path}: {exc}",
            )
