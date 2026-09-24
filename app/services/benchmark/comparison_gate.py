from __future__ import annotations

from app.domain.benchmark_report import BenchmarkReport


class IncompatibleBenchmarkComparisonError(ValueError):
    """Raised when two BenchmarkReports cannot be legitimately compared due to hard compatibility gate violations."""


def validate_benchmark_comparability(
    report_a: BenchmarkReport,
    report_b: BenchmarkReport,
) -> tuple[bool, str | None]:
    """
    Validates whether two BenchmarkReports satisfy strict compatibility gates
    to permit scientifically valid strategy comparison.

    Gates:
    1. Identical suite_key
    2. Identical suite_version
    3. Identical suite_fingerprint
    4. Identical case_count (same evaluated case set)
    5. Identical metric_definition_set_version
    6. Identical execution_mode (reject comparing OFFLINE against REAL)
    """
    if report_a.suite_key != report_b.suite_key:
        return (
            False,
            f"Suite key mismatch: baseline uses '{report_a.suite_key}', candidate uses '{report_b.suite_key}'",
        )

    if report_a.suite_version != report_b.suite_version:
        return (
            False,
            f"Suite version mismatch: baseline uses '{report_a.suite_version}', candidate uses '{report_b.suite_version}'",
        )

    if report_a.suite_fingerprint != report_b.suite_fingerprint:
        return (
            False,
            (
                f"Suite fingerprint mismatch: baseline has '{report_a.suite_fingerprint[:16]}...', "
                f"candidate has '{report_b.suite_fingerprint[:16]}...'"
            ),
        )

    if report_a.case_count != report_b.case_count:
        return (
            False,
            (
                f"Case count mismatch: baseline evaluated {report_a.case_count} cases, "
                f"candidate evaluated {report_b.case_count} cases"
            ),
        )

    if report_a.metric_definition_set_version != report_b.metric_definition_set_version:
        return (
            False,
            (
                f"Metric definition version mismatch: baseline uses '{report_a.metric_definition_set_version}', "
                f"candidate uses '{report_b.metric_definition_set_version}'"
            ),
        )

    if report_a.execution_mode != report_b.execution_mode:
        return (
            False,
            (
                f"Execution mode mismatch: baseline executed in {report_a.execution_mode.value}, "
                f"candidate executed in {report_b.execution_mode.value}. Cannot compare OFFLINE with REAL runs."
            ),
        )

    return (True, None)


def assert_benchmark_comparability(
    report_a: BenchmarkReport,
    report_b: BenchmarkReport,
) -> None:
    """Raises IncompatibleBenchmarkComparisonError if hard gates fail."""
    is_valid, reason = validate_benchmark_comparability(report_a, report_b)
    if not is_valid:
        raise IncompatibleBenchmarkComparisonError(reason or "Benchmark reports are incompatible")
