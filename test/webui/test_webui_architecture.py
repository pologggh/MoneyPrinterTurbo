"""Architecture compliance test for WebUI Agent Page and API Client.

Enforces strict architectural boundary:
webui/agent_page.py and webui/api_client.py must NEVER import directly from:
- app.persistence
- app.domain
- app.application
- app.services
- sqlalchemy

All backend communication must be strictly via HTTP API.
"""

import ast
from pathlib import Path
import pytest

FORBIDDEN_MODULE_PREFIXES = (
    "app.persistence",
    "app.domain",
    "app.application",
    "app.services",
    "sqlalchemy",
)

WEBUI_DIR = Path(__file__).resolve().parent.parent.parent / "webui"


def _extract_imported_modules(file_path: Path) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
    return imported


def test_api_client_zero_forbidden_imports():
    api_client_path = WEBUI_DIR / "api_client.py"
    assert api_client_path.exists(), f"File not found: {api_client_path}"

    imported = _extract_imported_modules(api_client_path)
    violations = [
        mod for mod in imported
        if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_MODULE_PREFIXES)
    ]
    assert not violations, f"webui/api_client.py violates architecture boundaries by importing: {violations}"


def test_agent_page_zero_forbidden_imports():
    agent_page_path = WEBUI_DIR / "agent_page.py"
    assert agent_page_path.exists(), f"File not found: {agent_page_path}"

    imported = _extract_imported_modules(agent_page_path)
    violations = [
        mod for mod in imported
        if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_MODULE_PREFIXES)
    ]
    assert not violations, f"webui/agent_page.py violates architecture boundaries by importing: {violations}"


def test_api_client_methods_interface():
    from webui.api_client import KnowledgeVideoApiClient

    expected_methods = [
        "create_task",
        "list_tasks",
        "get_task",
        "approve_task",
        "retry_task",
        "cancel_task",
        "authorize_research",
        "register_evidence",
        "process_knowledge",
        "retrieve_knowledge",
        "list_task_retrievals",
        "list_task_chunks",
        "get_task_events",
        "get_task_artifacts",
        "revise_task",
        "get_delivery_manifest",
        "get_download_url",
        "download_delivery_file",
    ]
    client = KnowledgeVideoApiClient(base_url="http://mock.test/api/v1")
    for method in expected_methods:
        assert hasattr(client, method) and callable(getattr(client, method)), f"Missing method: {method}"


def test_agent_page_render_function_exists():
    from webui import agent_page

    assert hasattr(agent_page, "render_agent_page")
    assert callable(agent_page.render_agent_page)
