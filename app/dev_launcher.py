from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from loguru import logger

from app.persistence.database_lifecycle import (
    init_database_on_startup,
    mask_database_url,
)
from app.persistence.session import get_database_url


@dataclass
class ServiceProcess:
    name: str
    command: list[str]
    log_path: Path
    proc: subprocess.Popen | None = None
    log_file: TextIO | None = None


def is_port_bound(host: str, port: int) -> bool:
    """Checks if a TCP port is already bound on the host."""
    bind_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.bind((bind_host, port))
            return False
        except OSError:
            return True


def wait_for_http_ping(host: str, port: int, timeout: float = 15.0) -> bool:
    """Polls http://<host>:<port>/ping until it responds 200 'pong' or times out."""
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{connect_host}:{port}/ping"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MPT-DevLauncher"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8", errors="ignore").strip()
                    if "pong" in body:
                        return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def get_log_tail(log_path: Path, lines: int = 10) -> str:
    """Reads the last N lines of a log file if available."""
    if not log_path.is_file():
        return "(no log file created)"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = "".join(all_lines[-lines:]).strip()
        return tail or "(empty log)"
    except Exception as exc:
        return f"(unable to read log: {exc})"


class DevLauncher:
    """
    Unified local development orchestrator.
    Manages FastAPI Backend, StageWorker, and Streamlit WebUI as independent OS processes.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        backend_host: str = "127.0.0.1",
        backend_port: int = 8080,
        webui_host: str = "127.0.0.1",
        webui_port: int = 8501,
        include_worker: bool = True,
        include_webui: bool = True,
    ) -> None:
        self.project_root = (
            project_root.resolve()
            if project_root
            else Path(__file__).resolve().parent.parent
        )
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.webui_host = webui_host
        self.webui_port = webui_port
        self.include_worker = include_worker
        self.include_webui = include_webui

        self.services: dict[str, ServiceProcess] = {}
        self._is_shutting_down = False

    def preflight_checks(self) -> tuple[bool, str]:
        """
        Runs preflight checks:
        1. Verifies database connectivity and runs pending migrations.
        2. Checks that backend and webui ports are not currently occupied.
        """
        # 1. Database check
        logger.info(
            f"Preflight: verifying database readiness and migrations against {mask_database_url(get_database_url())}..."
        )
        if not init_database_on_startup(timeout=10.0):
            return False, f"Database at {mask_database_url(get_database_url())} is not reachable or migration failed."

        # 2. Port conflict check for Backend
        if is_port_bound(self.backend_host, self.backend_port):
            return (
                False,
                f"Backend port {self.backend_port} on {self.backend_host} is already in use. "
                f"Check if an existing MoneyPrinterTurbo backend is already running.",
            )

        # 3. Port conflict check for WebUI
        if self.include_webui and is_port_bound(self.webui_host, self.webui_port):
            return (
                False,
                f"WebUI port {self.webui_port} on {self.webui_host} is already in use. "
                f"Please specify a different WebUI port or terminate the existing process.",
            )

        return True, "Preflight checks passed."

    def build_service_definitions(self) -> dict[str, ServiceProcess]:
        """Builds command and log definitions for independent processes."""
        py_exe = sys.executable
        services: dict[str, ServiceProcess] = {}

        # Backend
        services["backend"] = ServiceProcess(
            name="Backend",
            command=[py_exe, "main.py"],
            log_path=self.project_root / ".runtime-backend.log",
        )

        # StageWorker
        if self.include_worker:
            services["worker"] = ServiceProcess(
                name="StageWorker",
                command=[py_exe, "-m", "app.workers.stage_worker"],
                log_path=self.project_root / ".runtime-worker.log",
            )

        # WebUI
        if self.include_webui:
            webui_main = str(self.project_root / "webui" / "Main.py")
            services["webui"] = ServiceProcess(
                name="WebUI",
                command=[
                    py_exe,
                    "-m",
                    "streamlit",
                    "run",
                    webui_main,
                    f"--server.address={self.webui_host}",
                    f"--server.port={self.webui_port}",
                    f"--browser.serverAddress={self.webui_host}",
                    "--server.enableCORS=True",
                    "--browser.gatherUsageStats=False",
                    "--client.toolbarMode=minimal",
                    "--logger.hideWelcomeMessage=True",
                    "--server.showEmailPrompt=False",
                ],
                log_path=self.project_root / ".runtime-webui.log",
            )

        return services

    def start(self) -> bool:
        """
        Starts all configured services as independent OS processes.
        Returns True if all started and passed readiness checks, False otherwise.
        """
        ok, reason = self.preflight_checks()
        if not ok:
            logger.error(f"Startup aborted: {reason}")
            print(f"\n[ERROR] Startup aborted: {reason}\n")
            return False

        self.services = self.build_service_definitions()

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        env["DATABASE_URL"] = get_database_url()
        env["MPT_WEBUI_HOST"] = self.webui_host
        env["MPT_WEBUI_PORT"] = str(self.webui_port)
        connect_host = "127.0.0.1" if self.backend_host in ("0.0.0.0", "") else self.backend_host
        env["KNOWLEDGE_VIDEO_API_BASE_URL"] = f"http://{connect_host}:{self.backend_port}/api/v1"

        print("\n" + "=" * 65)
        print(" MoneyPrinterTurbo Unified Development Environment")
        print("=" * 65)
        print(f" Database: {mask_database_url(get_database_url())}")
        print(f" Root:     {self.project_root}")
        print("=" * 65)

        for key, srv in self.services.items():
            try:
                srv.log_file = open(srv.log_path, "w", encoding="utf-8", errors="replace")
                srv.proc = subprocess.Popen(
                    srv.command,
                    cwd=str(self.project_root),
                    env=env,
                    stdout=srv.log_file,
                    stderr=subprocess.STDOUT,
                )
                logger.info(f"Spawned {srv.name} (PID: {srv.proc.pid})")
            except Exception as exc:
                logger.error(f"Failed to spawn {srv.name}: {exc}")
                print(f"[FAILED] Could not spawn {srv.name}: {exc}")
                self.shutdown()
                return False

        # Verify Backend readiness
        print("\nWaiting for Backend health check (/ping)...")
        if not wait_for_http_ping(self.backend_host, self.backend_port, timeout=20.0):
            backend_proc = self.services["backend"].proc
            exit_code = backend_proc.poll() if backend_proc else None
            tail = get_log_tail(self.services["backend"].log_path)
            print(f"[FAILED] Backend failed to become healthy (exit code: {exit_code}).")
            print(f"--- Tail of {self.services['backend'].log_path.name} ---")
            print(tail)
            print("-" * 50)
            self.shutdown()
            return False

        # Verify Worker stayed alive (if started)
        if self.include_worker and "worker" in self.services:
            time.sleep(1.0)
            worker_proc = self.services["worker"].proc
            if worker_proc and worker_proc.poll() is not None:
                tail = get_log_tail(self.services["worker"].log_path)
                print(f"[FAILED] StageWorker exited prematurely with code {worker_proc.poll()}.")
                print(f"--- Tail of {self.services['worker'].log_path.name} ---")
                print(tail)
                print("-" * 50)
                self.shutdown()
                return False

        # Verify WebUI stayed alive (if started)
        if self.include_webui and "webui" in self.services:
            time.sleep(1.0)
            webui_proc = self.services["webui"].proc
            if webui_proc and webui_proc.poll() is not None:
                tail = get_log_tail(self.services["webui"].log_path)
                print(f"[FAILED] WebUI exited prematurely with code {webui_proc.poll()}.")
                print(f"--- Tail of {self.services['webui'].log_path.name} ---")
                print(tail)
                print("-" * 50)
                self.shutdown()
                return False

        self.print_status_table()
        return True

    def print_status_table(self) -> None:
        """Prints a clean, structured status table for developers."""
        print("\n" + "-" * 65)
        print(f" {'SERVICE':<14} {'STATUS':<10} {'PID':<8} {'DETAILS'}")
        print("-" * 65)
        print(f" {'Database':<14} {'READY':<10} {'-':<8} {mask_database_url(get_database_url())}")

        connect_host = "127.0.0.1" if self.backend_host in ("0.0.0.0", "") else self.backend_host
        if "backend" in self.services and self.services["backend"].proc:
            pid = str(self.services["backend"].proc.pid)
            print(f" {'Backend':<14} {'RUNNING':<10} {pid:<8} http://{connect_host}:{self.backend_port} (ping: OK)")

        if "worker" in self.services and self.services["worker"].proc:
            pid = str(self.services["worker"].proc.pid)
            print(f" {'StageWorker':<14} {'RUNNING':<10} {pid:<8} independent process (polling loop)")

        if "webui" in self.services and self.services["webui"].proc:
            pid = str(self.services["webui"].proc.pid)
            print(f" {'WebUI':<14} {'RUNNING':<10} {pid:<8} http://{self.webui_host}:{self.webui_port}")

        print("-" * 65)
        print(" Logs:")
        for srv in self.services.values():
            print(f"   {srv.name:<12} -> {srv.log_path.name}")
        print("=" * 65)
        print(" Press Ctrl+C to stop all services cleanly.\n")

    def run_supervision_loop(self) -> None:
        """Supervises running processes until Ctrl+C or a service failure occurs."""
        while not self._is_shutting_down:
            for srv in self.services.values():
                if srv.proc is not None:
                    code = srv.proc.poll()
                    if code is not None:
                        logger.warning(f"Service '{srv.name}' (PID {srv.proc.pid}) exited with code {code}.")
                        print(f"\n[ALERT] Service '{srv.name}' exited unexpectedly with code {code}!")
                        tail = get_log_tail(srv.log_path, lines=15)
                        print(f"--- Tail of {srv.log_path.name} ---")
                        print(tail)
                        print("-" * 50)
                        self.shutdown()
                        return
            time.sleep(1.0)

    def shutdown(self) -> None:
        """
        Gracefully stops ONLY the child processes created by this launcher instance.
        Never calls global process killers.
        """
        if self._is_shutting_down:
            return
        self._is_shutting_down = True

        print("\nStopping services gracefully...")
        for srv in self.services.values():
            if srv.proc is not None and srv.proc.poll() is None:
                logger.info(f"Stopping {srv.name} (PID: {srv.proc.pid})...")
                try:
                    srv.proc.terminate()
                except Exception as exc:
                    logger.warning(f"Error terminating {srv.name}: {exc}")

        # Wait up to 3 seconds for graceful shutdown
        for srv in self.services.values():
            if srv.proc is not None and srv.proc.poll() is None:
                try:
                    srv.proc.wait(timeout=3.0)
                    logger.info(f"{srv.name} stopped cleanly.")
                except subprocess.TimeoutExpired:
                    logger.warning(f"{srv.name} did not exit within timeout, killing...")
                    try:
                        srv.proc.kill()
                    except Exception:
                        pass

        # Close log file handles
        for srv in self.services.values():
            if srv.log_file is not None:
                is_closed = getattr(srv.log_file, "closed", False)
                if not is_closed:
                    try:
                        srv.log_file.close()
                    except Exception:
                        pass

        print("All launcher services stopped. No orphan processes remain.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MoneyPrinterTurbo Unified Dev Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Backend listen host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Backend listen port (default: 8080)")
    parser.add_argument("--webui-host", type=str, default="127.0.0.1", help="WebUI listen host (default: 127.0.0.1)")
    parser.add_argument("--webui-port", type=int, default=8501, help="WebUI listen port (default: 8501)")
    parser.add_argument("--no-worker", action="store_true", help="Do not start StageWorker process")
    parser.add_argument("--no-webui", action="store_true", help="Do not start Streamlit WebUI process")
    parser.add_argument("--check-only", action="store_true", help="Only run preflight checks and exit")
    args = parser.parse_args()

    launcher = DevLauncher(
        backend_host=args.host,
        backend_port=args.port,
        webui_host=args.webui_host,
        webui_port=args.webui_port,
        include_worker=not args.no_worker,
        include_webui=not args.no_webui,
    )

    if args.check_only:
        ok, msg = launcher.preflight_checks()
        if ok:
            print(f"[OK] {msg}")
            sys.exit(0)
        else:
            print(f"[FAILED] {msg}")
            sys.exit(1)

    def _sig_handler(sig, frame):
        print("\nReceived stop signal...")
        launcher.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    if not launcher.start():
        sys.exit(1)

    try:
        launcher.run_supervision_loop()
    except KeyboardInterrupt:
        launcher.shutdown()


if __name__ == "__main__":
    main()
