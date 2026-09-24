from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.dev_launcher import (
    DevLauncher,
    ServiceProcess,
    get_log_tail,
    is_port_bound,
    wait_for_http_ping,
)


def test_dev_launcher_build_service_definitions_has_independent_processes():
    launcher = DevLauncher(
        backend_host="127.0.0.1",
        backend_port=8080,
        webui_host="127.0.0.1",
        webui_port=8501,
        include_worker=True,
        include_webui=True,
    )
    services = launcher.build_service_definitions()

    assert "backend" in services
    assert "worker" in services
    assert "webui" in services

    # Ensure separate commands and independent processes
    backend_cmd = services["backend"].command
    worker_cmd = services["worker"].command
    webui_cmd = services["webui"].command

    assert backend_cmd[0] == sys.executable
    assert "main.py" in backend_cmd

    assert worker_cmd[0] == sys.executable
    assert "-m" in worker_cmd
    assert "app.workers.stage_worker" in worker_cmd

    assert webui_cmd[0] == sys.executable
    assert "-m" in webui_cmd
    assert "streamlit" in webui_cmd
    assert any("Main.py" in arg for arg in webui_cmd)

    # Distinct log files
    assert services["backend"].log_path != services["worker"].log_path
    assert services["backend"].log_path != services["webui"].log_path
    assert services["worker"].log_path != services["webui"].log_path


def test_dev_launcher_conditional_service_inclusion():
    launcher_no_worker = DevLauncher(include_worker=False, include_webui=True)
    services_no_worker = launcher_no_worker.build_service_definitions()
    assert "backend" in services_no_worker
    assert "worker" not in services_no_worker
    assert "webui" in services_no_worker

    launcher_no_webui = DevLauncher(include_worker=True, include_webui=False)
    services_no_webui = launcher_no_webui.build_service_definitions()
    assert "backend" in services_no_webui
    assert "worker" in services_no_webui
    assert "webui" not in services_no_webui


def test_dev_launcher_preflight_detects_db_failure():
    launcher = DevLauncher()
    with patch("app.dev_launcher.init_database_on_startup", return_value=False):
        ok, reason = launcher.preflight_checks()
        assert not ok
        assert "Database at" in reason
        assert "not reachable" in reason


def test_dev_launcher_preflight_detects_backend_port_conflict():
    launcher = DevLauncher(backend_port=8080)
    with (
        patch("app.dev_launcher.init_database_on_startup", return_value=True),
        patch("app.dev_launcher.is_port_bound", side_effect=lambda h, p: p == 8080),
    ):
        ok, reason = launcher.preflight_checks()
        assert not ok
        assert "Backend port 8080" in reason
        assert "already in use" in reason


def test_dev_launcher_preflight_detects_webui_port_conflict():
    launcher = DevLauncher(backend_port=8080, webui_port=8501, include_webui=True)
    with (
        patch("app.dev_launcher.init_database_on_startup", return_value=True),
        patch("app.dev_launcher.is_port_bound", side_effect=lambda h, p: p == 8501),
    ):
        ok, reason = launcher.preflight_checks()
        assert not ok
        assert "WebUI port 8501" in reason
        assert "already in use" in reason


def test_dev_launcher_start_backend_failure_shuts_down():
    launcher = DevLauncher(include_worker=True, include_webui=True)
    mock_backend_proc = MagicMock(spec=subprocess.Popen)
    mock_backend_proc.pid = 1001
    mock_backend_proc.poll.return_value = None

    mock_worker_proc = MagicMock(spec=subprocess.Popen)
    mock_worker_proc.pid = 1002
    mock_worker_proc.poll.return_value = None

    mock_webui_proc = MagicMock(spec=subprocess.Popen)
    mock_webui_proc.pid = 1003
    mock_webui_proc.poll.return_value = None

    mock_procs = [mock_backend_proc, mock_worker_proc, mock_webui_proc]

    with (
        patch.object(launcher, "preflight_checks", return_value=(True, "OK")),
        patch("builtins.open", MagicMock()),
        patch("subprocess.Popen", side_effect=mock_procs),
        patch("app.dev_launcher.wait_for_http_ping", return_value=False),
        patch.object(launcher, "shutdown") as mock_shutdown,
    ):
        result = launcher.start()
        assert result is False
        mock_shutdown.assert_called_once()


def test_dev_launcher_start_worker_early_exit_shuts_down():
    launcher = DevLauncher(include_worker=True, include_webui=True)
    mock_backend_proc = MagicMock(spec=subprocess.Popen)
    mock_backend_proc.pid = 1001
    mock_backend_proc.poll.return_value = None

    mock_worker_proc = MagicMock(spec=subprocess.Popen)
    mock_worker_proc.pid = 1002
    # Simulate worker crashing on startup
    mock_worker_proc.poll.return_value = 1

    mock_webui_proc = MagicMock(spec=subprocess.Popen)
    mock_webui_proc.pid = 1003
    mock_webui_proc.poll.return_value = None

    mock_procs = [mock_backend_proc, mock_worker_proc, mock_webui_proc]

    with (
        patch.object(launcher, "preflight_checks", return_value=(True, "OK")),
        patch("builtins.open", MagicMock()),
        patch("subprocess.Popen", side_effect=mock_procs),
        patch("app.dev_launcher.wait_for_http_ping", return_value=True),
        patch.object(launcher, "shutdown") as mock_shutdown,
    ):
        result = launcher.start()
        assert result is False
        mock_shutdown.assert_called_once()


def test_dev_launcher_shutdown_terminates_only_own_children():
    launcher = DevLauncher()
    mock_p1 = MagicMock(spec=subprocess.Popen)
    mock_p1.pid = 1111
    mock_p1.poll.return_value = None

    mock_p2 = MagicMock(spec=subprocess.Popen)
    mock_p2.pid = 2222
    mock_p2.poll.return_value = None

    mock_f1 = MagicMock()
    mock_f1.closed = False
    mock_f2 = MagicMock()
    mock_f2.closed = False

    launcher.services = {
        "srv1": ServiceProcess(name="S1", command=["cmd1"], log_path=Path("s1.log"), proc=mock_p1, log_file=mock_f1),
        "srv2": ServiceProcess(name="S2", command=["cmd2"], log_path=Path("s2.log"), proc=mock_p2, log_file=mock_f2),
    }

    launcher.shutdown()

    mock_p1.terminate.assert_called_once()
    mock_p2.terminate.assert_called_once()
    mock_p1.wait.assert_called_once()
    mock_p2.wait.assert_called_once()
    mock_f1.close.assert_called_once()
    mock_f2.close.assert_called_once()


def test_dev_launcher_supervision_loop_handles_child_exit():
    launcher = DevLauncher()
    mock_p1 = MagicMock(spec=subprocess.Popen)
    mock_p1.pid = 1111
    # First check: alive. Next check: exited with error code 42
    mock_p1.poll.side_effect = [None, 42]

    launcher.services = {
        "backend": ServiceProcess(name="Backend", command=["main.py"], log_path=Path("b.log"), proc=mock_p1),
    }

    with (
        patch("time.sleep", return_value=None),
        patch.object(launcher, "shutdown") as mock_shutdown,
    ):
        launcher.run_supervision_loop()
        mock_shutdown.assert_called_once()


def test_get_log_tail_handles_missing_and_existing_files(tmp_path):
    missing_file = tmp_path / "non_existent.log"
    assert get_log_tail(missing_file) == "(no log file created)"

    log_file = tmp_path / "test.log"
    lines = [f"Line {i}\n" for i in range(1, 20)]
    log_file.write_text("".join(lines), encoding="utf-8")

    tail = get_log_tail(log_file, lines=3)
    assert "Line 17" in tail
    assert "Line 18" in tail
    assert "Line 19" in tail
    assert "Line 10" not in tail
