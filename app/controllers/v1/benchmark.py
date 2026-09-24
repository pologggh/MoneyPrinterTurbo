from fastapi import Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from app.controllers import base
from app.controllers.v1.base import new_router
from app.domain.benchmark_metrics import METRIC_SET_VERSION_V1
from app.persistence.session import get_session
from app.services.benchmark.comparison_gate import IncompatibleBenchmarkComparisonError
from app.services.benchmark.report_service import BenchmarkReportService
from app.utils import utils

router = new_router(dependencies=[Depends(base.verify_token)])


class ApiGenerateReportRequest(BaseModel):
    metric_version: str = METRIC_SET_VERSION_V1


class ApiCompareReportsRequest(BaseModel):
    baseline_report_id: str | None = None
    candidate_report_id: str | None = None
    baseline_run_id: str | None = None
    candidate_run_id: str | None = None
    metric_version: str = METRIC_SET_VERSION_V1


@router.post(
    "/benchmark/runs/{run_id}/report",
    summary="Generate or retrieve benchmark metrics report for a run",
)
def generate_benchmark_report(
    run_id: str,
    body: ApiGenerateReportRequest | None = None,
):
    metric_version = body.metric_version if body else METRIC_SET_VERSION_V1
    report_service = BenchmarkReportService(get_session)
    try:
        report = report_service.generate_report(
            benchmark_run_id=run_id,
            metric_version=metric_version,
        )
        return utils.get_response(200, report.to_dict())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        logger.exception(f"Benchmark report generation failed for run {run_id}")
        raise HTTPException(
            status_code=500,
            detail="Benchmark report generation failed.",
        )


@router.get(
    "/benchmark/reports/{report_id}",
    summary="Retrieve an existing benchmark metrics report",
)
def get_benchmark_report(report_id: str):
    report_service = BenchmarkReportService(get_session)
    report = report_service.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"BenchmarkReport '{report_id}' not found")
    return utils.get_response(200, report.to_dict())


@router.post(
    "/benchmark/compare",
    summary="Compare two benchmark runs or reports and produce comparison report",
)
def compare_benchmarks(body: ApiCompareReportsRequest):
    report_service = BenchmarkReportService(get_session)
    try:
        if body.baseline_report_id and body.candidate_report_id:
            comp = report_service.compare_reports(
                baseline_report_id=body.baseline_report_id,
                candidate_report_id=body.candidate_report_id,
            )
        elif body.baseline_run_id and body.candidate_run_id:
            comp = report_service.compare_runs(
                baseline_run_id=body.baseline_run_id,
                candidate_run_id=body.candidate_run_id,
                metric_version=body.metric_version,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either (baseline_report_id, candidate_report_id) or (baseline_run_id, candidate_run_id)",
            )
        return utils.get_response(200, comp.to_dict())
    except IncompatibleBenchmarkComparisonError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        logger.exception("Benchmark report comparison failed")
        raise HTTPException(
            status_code=500,
            detail="Benchmark report comparison failed.",
        )


@router.get(
    "/benchmark/comparisons/{comparison_id}",
    summary="Retrieve an existing benchmark comparison report",
)
def get_benchmark_comparison(comparison_id: str):
    report_service = BenchmarkReportService(get_session)
    comp = report_service.get_comparison_report(comparison_id)
    if not comp:
        raise HTTPException(status_code=404, detail=f"BenchmarkComparisonReport '{comparison_id}' not found")
    return utils.get_response(200, comp.to_dict())


@router.get(
    "/benchmark/reports/{report_id}/export",
    summary="Export benchmark report as deterministic JSON",
)
def export_benchmark_report_json(report_id: str):
    report_service = BenchmarkReportService(get_session)
    report = report_service.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"BenchmarkReport '{report_id}' not found")
    return utils.get_response(200, report.to_dict())


@router.get(
    "/benchmark/comparisons/{comparison_id}/export",
    summary="Export benchmark comparison report as deterministic JSON",
)
def export_benchmark_comparison_json(comparison_id: str):
    report_service = BenchmarkReportService(get_session)
    comp = report_service.get_comparison_report(comparison_id)
    if not comp:
        raise HTTPException(status_code=404, detail=f"BenchmarkComparisonReport '{comparison_id}' not found")
    return utils.get_response(200, comp.to_dict())
