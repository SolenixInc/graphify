"""Tests for Graphify's caller-aware Codex CLI backend.

The Codex harness is mocked so these tests prove routing and JSONL parsing without
sending a corpus or requiring an authenticated live session.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from graphify import llm


_GRAPH_FRAGMENT = {
    "nodes": [
        {"id": "foo_module", "label": "Foo", "file_type": "document", "source_file": "foo.md"},
    ],
    "edges": [],
    "hyperedges": [],
}


def _codex_jsonl(message: str) -> str:
    return "\n".join((
        json.dumps({"type": "thread.started", "thread_id": "thread_123"}),
        json.dumps({"type": "item.completed", "item": {"type": "error", "message": "ignored"}}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": message}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}}),
    ))


@pytest.fixture
def fake_codex(monkeypatch):
    completed = MagicMock(returncode=0, stdout=_codex_jsonl(json.dumps(_GRAPH_FRAGMENT)), stderr="")
    monkeypatch.setattr(llm, "_response_is_hollow", lambda raw, parsed: False)
    with patch("shutil.which", return_value="/fake/bin/codex"), \
         patch("subprocess.run", return_value=completed) as run:
        yield run


def test_codex_cli_returns_parsed_graph_fragment(fake_codex):
    result = llm._call_codex_cli("dummy source")

    assert result["nodes"] == _GRAPH_FRAGMENT["nodes"]
    assert result["edges"] == []
    assert result["hyperedges"] == []


def test_codex_cli_uses_lean_jsonl_ephemeral_and_schema_output(fake_codex):
    llm._call_codex_cli("UNIQUE_SOURCE_MARKER")

    argv = fake_codex.call_args.args[0]
    assert argv[:4] == ["codex", "exec", "--json", "--ephemeral"]
    assert "--ignore-user-config" in argv
    assert "--output-schema" in argv
    assert argv[-1] == "-"
    sent = fake_codex.call_args.kwargs["input"]
    assert "UNIQUE_SOURCE_MARKER" in sent
    assert "graphify semantic extraction agent" in sent


def test_codex_cli_refuses_missing_harness():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Codex CLI not found"):
            llm._call_codex_cli("dummy")


def test_caller_selection_prefers_codex_harness(monkeypatch):
    monkeypatch.setenv("CODEX_THREAD_ID", "thread_123")
    monkeypatch.delenv("CLAUDECODE", raising=False)
    with patch("shutil.which", return_value="/fake/bin/codex"):
        assert llm.detect_backend() == "codex-cli"


def test_caller_selection_prefers_claude_harness(monkeypatch):
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.setenv("CLAUDECODE", "1")
    with patch("shutil.which", return_value="/fake/bin/claude"):
        assert llm.detect_backend() == "claude-cli"


def test_caller_selection_fails_closed_when_codex_is_unavailable(monkeypatch):
    monkeypatch.setenv("CODEX_THREAD_ID", "thread_123")
    monkeypatch.delenv("CLAUDECODE", raising=False)
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Codex session detected"):
            llm.detect_backend()


def test_explicit_backend_overrides_caller_selection(monkeypatch):
    monkeypatch.setenv("CODEX_THREAD_ID", "thread_123")
    with patch("shutil.which", return_value="/fake/bin/codex"):
        assert llm.resolve_backend("gemini") == "gemini"


def test_codex_backend_is_registered_with_zero_cost():
    assert "codex-cli" in llm.BACKENDS
    assert llm.estimate_cost("codex-cli", 1_000_000, 1_000_000) == 0.0


def test_codex_schema_is_strict_for_every_object():
    schema = json.loads(llm._CODEX_EXTRACTION_JSON_SCHEMA)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"nodes", "edges", "hyperedges"}
    for key in ("nodes", "edges", "hyperedges"):
        item_schema = schema["properties"][key]["items"]
        assert item_schema["additionalProperties"] is False
        assert set(item_schema["required"]) == set(item_schema["properties"])
