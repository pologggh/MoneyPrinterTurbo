from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from app.domain.benchmark_comparison import BenchmarkComparisonReport
from app.domain.benchmark_metrics import METRIC_SET_VERSION_V1
from app.domain.benchmark_report import BenchmarkReport
from app.persistence.repositories import BenchmarkRepository
from app.services.benchmark.comparison_service import BenchmarkComparisonService
from app.services.benchmark.metric_extractor import BenchmarkMetricExtractor


class BenchmarkReportService:
    """
    Service orchestrating immutable BenchmarkReport generation and
    BenchmarkComparisonReport creation.
    """

    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def generate_report(
        self,
        benchmark_run_id: str,
        metric_version: str = METRIC_SET_VERSION_V1,
    ) -> BenchmarkReport:
        """
        Generates and persists an immutable BenchmarkReport derived from historical records.
        If a report already exists for this run and metric_version, returns the existing record.
        """
        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            existing = repo.get_report_by_run(benchmark_run_id, metric_version)
            if existing:
                return existing

            run = repo.get_run(benchmark_run_id)
            if not run:
                raise ValueError(f"BenchmarkRun '{benchmark_run_id}' not found.")

            case_results = repo.list_case_results_for_run(benchmark_run_id)
            extractor = BenchmarkMetricExtractor(session)
            (
                metrics,
                latencies,
                cost_summary,
                usage,
                case_count,
                shot_count,
            ) = extractor.extract_metrics_for_run(run, case_results)

            report = BenchmarkReport(
                benchmark_run_id=run.benchmark_run_id,
                suite_key=run.suite_key,
                suite_version=run.suite_version,
                suite_fingerprint=run.suite_fingerprint,
                variant=run.variant,
                execution_mode=run.execution_mode,
                metrics=metrics,
                stage_latencies=latencies,
                cost_summary=cost_summary,
                provider_model_usage=usage,
                case_count=case_count,
                shot_count=shot_count,
                metric_definition_set_version=metric_version,
            )

            saved = repo.add_report(report)
            session.commit()
            return saved
        finally:
            session.close()

    def get_report(self, report_id: str) -> BenchmarkReport | None:
        """Retrieves a BenchmarkReport by report_id."""
        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            return repo.get_report(report_id)
        finally:
            session.close()

    def get_report_by_run(
        self,
        benchmark_run_id: str,
        metric_version: str = METRIC_SET_VERSION_V1,
    ) -> BenchmarkReport | None:
        """Retrieves a BenchmarkReport for a specific run and metric version."""
        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            return repo.get_report_by_run(benchmark_run_id, metric_version)
        finally:
            session.close()

    def compare_reports(
        self,
        baseline_report_id: str,
        candidate_report_id: str,
    ) -> BenchmarkComparisonReport:
        """
        Validates compatibility and computes an immutable BenchmarkComparisonReport
        between two persisted BenchmarkReports.
        """
        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            b_rep = repo.get_report(baseline_report_id)
            if not b_rep:
                raise ValueError(f"Baseline BenchmarkReport '{baseline_report_id}' not found.")

            c_rep = repo.get_report(candidate_report_id)
            if not c_rep:
                raise ValueError(f"Candidate BenchmarkReport '{candidate_report_id}' not found.")

            comp_svc = BenchmarkComparisonService(bench_repo=repo)
            comp = comp_svc.compare_reports(b_rep, c_rep)

            saved = repo.add_comparison_report(comp)
            session.commit()
            return saved
        finally:
            session.close()

    def compare_runs(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        metric_version: str = METRIC_SET_VERSION_V1,
    ) -> BenchmarkComparisonReport:
        """
        Convenience method: generates reports for both runs if needed,
        then calculates and persists the comparison report.
        """
        b_rep = self.generate_report(baseline_run_id, metric_version)
        c_rep = self.generate_report(candidate_run_id, metric_version)
        return self.compare_reports(b_rep.report_id, c_rep.report_id)

    def get_comparison_report(self, comparison_id: str) -> BenchmarkComparisonReport | None:
        """Retrieves a BenchmarkComparisonReport by comparison_id."""
        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            return repo.get_comparison_report(comparison_id)
        finally:
            session.close()
