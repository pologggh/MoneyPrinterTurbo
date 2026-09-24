from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.domain.benchmark import (
    BenchmarkCase,
    BenchmarkCaseRef,
    BenchmarkExpectedConstraints,
    BenchmarkSuite,
    KnowledgeFixtureRef,
)
from app.domain.enums import BeatType


def get_default_benchmark_dir() -> Path:
    """Returns root path to fixtures/benchmark/."""
    # Assuming app/services/benchmark/dataset_loader.py -> project root is 3 levels up
    return Path(__file__).resolve().parent.parent.parent.parent / "fixtures" / "benchmark"


def load_knowledge_fixture(
    file_path: str | Path,
    fixture_id: str = "",
    fixture_version: str = "v1.0",
    base_dir: Path | None = None,
) -> KnowledgeFixtureRef:
    """Loads a knowledge fixture file and calculates its SHA-256 hash."""
    p = Path(file_path)
    if not p.is_absolute() and base_dir is not None:
        p = (base_dir / p).resolve()
    elif not p.is_absolute():
        p = (Path.cwd() / p).resolve()

    if not p.is_file():
        raise FileNotFoundError(f"Knowledge fixture file not found: {p}")

    content_bytes = p.read_bytes()
    sha256_hash = hashlib.sha256(content_bytes).hexdigest()

    fid = fixture_id or p.stem
    return KnowledgeFixtureRef(
        fixture_id=fid,
        fixture_version=fixture_version,
        file_path=str(p),
        content_sha256=sha256_hash,
    )


def load_benchmark_suite_from_json(
    suite_json_path: str | Path,
    base_dir: Path | None = None,
) -> tuple[BenchmarkSuite, dict[str, BenchmarkCase]]:
    """
    Parses a BenchmarkSuite JSON definition into domain objects.
    Returns the BenchmarkSuite domain model and a dictionary of case_key -> BenchmarkCase.
    """
    p = Path(suite_json_path)
    if not p.is_file():
        raise FileNotFoundError(f"Benchmark suite definition not found: {p}")

    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    suite_key = data["suite_key"]
    suite_version = data["suite_version"]

    root_dir = base_dir or p.resolve().parent.parent.parent

    case_refs: list[BenchmarkCaseRef] = []
    cases_dict: dict[str, BenchmarkCase] = {}

    for case_data in data.get("cases", []):
        ckey = case_data["case_key"]
        cver = case_data.get("case_version", "v1")

        fix_data = case_data["knowledge_fixture"]
        fix_rel_path = fix_data["file_path"]
        fix_abs_path = (root_dir / fix_rel_path).resolve()

        if not fix_abs_path.is_file() and Path(fix_rel_path).is_file():
            fix_abs_path = Path(fix_rel_path).resolve()

        actual_sha = ""
        if fix_abs_path.is_file():
            actual_sha = hashlib.sha256(fix_abs_path.read_bytes()).hexdigest()

        fix_ref = KnowledgeFixtureRef(
            fixture_id=fix_data["fixture_id"],
            fixture_version=fix_data.get("fixture_version", "v1.0"),
            file_path=str(fix_abs_path),
            content_sha256=fix_data.get("content_sha256") or actual_sha,
        )

        c_data = case_data.get("expected_constraints", {})
        req_types = tuple(
            BeatType(bt) if isinstance(bt, str) else bt
            for bt in c_data.get("required_beat_types", [])
        )
        constraints = BenchmarkExpectedConstraints(
            min_beats=c_data.get("min_beats", 1),
            max_beats=c_data.get("max_beats", 10),
            required_beat_types=req_types,
            require_evidence_for_knowledge_beats=c_data.get("require_evidence_for_knowledge_beats", True),
            target_duration_tolerance=c_data.get("target_duration_tolerance", 2.0),
            minimum_shots=c_data.get("minimum_shots"),
            maximum_shots=c_data.get("maximum_shots"),
        )

        bcase = BenchmarkCase(
            case_key=ckey,
            case_version=cver,
            title=case_data["title"],
            topic=case_data["topic"],
            target_duration=float(case_data["target_duration"]),
            user_instruction=case_data.get("user_instruction"),
            knowledge_fixture_ref=fix_ref,
            expected_constraints=constraints,
            tags=tuple(case_data.get("tags", [])),
        )
        cases_dict[ckey] = bcase

        cref = BenchmarkCaseRef(
            case_key=ckey,
            case_version=cver,
            content_fingerprint=bcase.content_fingerprint,
        )
        case_refs.append(cref)

    bsuite = BenchmarkSuite(
        suite_key=suite_key,
        suite_version=suite_version,
        benchmark_case_refs=tuple(case_refs),
    )

    return (bsuite, cases_dict)


def load_benchmark_suite(
    suite_key: str,
    suite_version: str,
    suites_dir: Path | None = None,
) -> tuple[BenchmarkSuite, dict[str, BenchmarkCase]]:
    """Loads a suite by key and version from the default suites directory."""
    sdir = suites_dir or (get_default_benchmark_dir() / "suites")
    # Convert hyphens to underscores for filename if needed
    candidates = [
        sdir / f"{suite_key}_{suite_version}.json",
        sdir / f"{suite_key.replace('-', '_')}_{suite_version}.json",
        sdir / f"{suite_key.replace('-', '_')}.json",
        sdir / f"{suite_key}.json",
    ]
    for c in candidates:
        if c.is_file():
            suite, cases = load_benchmark_suite_from_json(c)
            if suite.suite_key == suite_key and suite.suite_version == suite_version:
                return (suite, cases)

    raise FileNotFoundError(
        f"Benchmark suite '{suite_key}' (version: {suite_version}) not found in {sdir}"
    )


def validate_benchmark_dataset(suites_dir: Path | None = None) -> list[str]:
    """
    Validates all benchmark suite definitions and their referenced fixtures without making external/network calls.
    Returns a list of validation error strings. If empty, dataset is 100% valid.
    """
    errors: list[str] = []
    sdir = suites_dir or (get_default_benchmark_dir() / "suites")
    if not sdir.is_dir():
        return [f"Suites directory does not exist: {sdir}"]

    json_files = list(sdir.glob("*.json"))
    if not json_files:
        return [f"No suite JSON files found in {sdir}"]

    for jf in json_files:
        try:
            suite, cases = load_benchmark_suite_from_json(jf)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to load suite from {jf}: {exc}")
            continue

        seen_keys: set[str] = set()
        for cref in suite.benchmark_case_refs:
            if cref.case_key in seen_keys:
                errors.append(f"Suite '{suite.suite_key}' contains duplicate case_key: '{cref.case_key}'")
            seen_keys.add(cref.case_key)

            case = cases.get(cref.case_key)
            if not case:
                errors.append(f"Suite references missing case '{cref.case_key}'")
                continue

            # Validate target duration
            if case.target_duration <= 0:
                errors.append(f"Case '{case.case_key}' has non-positive duration: {case.target_duration}")

            # Validate fixture exists and sha256 matches
            fix_p = Path(case.knowledge_fixture_ref.file_path)
            if not fix_p.is_file():
                errors.append(f"Case '{case.case_key}' fixture file does not exist: {fix_p}")
            else:
                actual_hash = hashlib.sha256(fix_p.read_bytes()).hexdigest()
                if actual_hash != case.knowledge_fixture_ref.content_sha256:
                    errors.append(
                        f"Case '{case.case_key}' fixture SHA-256 mismatch! "
                        f"Expected: {case.knowledge_fixture_ref.content_sha256}, actual: {actual_hash}"
                    )

            # Validate constraints
            if case.expected_constraints.min_beats > case.expected_constraints.max_beats:
                errors.append(f"Case '{case.case_key}' has min_beats > max_beats")

    return errors
