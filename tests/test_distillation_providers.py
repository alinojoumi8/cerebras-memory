from __future__ import annotations

from dataclasses import replace
import json

import pytest

from distillation import (
    DISTILLATION_FIELDS,
    DeepSeekDistiller,
    DistillationUnavailable,
    create_distiller,
    sanitize_role_contradictions,
    validate_distillation,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _deepseek_settings(settings_factory):
    return replace(
        settings_factory().distillation,
        provider="deepseek",
        endpoint="https://api.deepseek.com/beta",
        model="deepseek-v4-flash",
        api_key_env="DEEPSEEK_API_KEY",
        prompt_version="agent-dialogue-v5-deepseek-isolated-json-fallback",
    )


def test_deepseek_uses_forced_strict_tool_schema(settings_factory, monkeypatch):
    settings = _deepseek_settings(settings_factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret-key")
    captured = {}
    structured = {
        "user_goal": "Finish the rollout",
        "summary": "The rollout was completed.",
        "outcome": "Complete",
        "decisions": ["Keep raw citations"],
        "artifacts": ["rollout.md"],
        "systems": ["Cerebras Memory"],
        "open_questions": [],
        "keywords": ["rollout"],
    }

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "record_distillation",
                                        "arguments": json.dumps(structured),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("distillation.urlopen", fake_urlopen)
    result = DeepSeekDistiller(settings).distill("USER [2026-07-21T00:00:00Z]\nhello")

    assert result == structured
    assert captured["url"] == "https://api.deepseek.com/beta/chat/completions"
    assert captured["authorization"] == "Bearer fixture-secret-key"
    payload = captured["payload"]
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["parallel_tool_calls"] is False
    assert payload["messages"][0]["role"] == "system"
    untrusted = json.loads(payload["messages"][1]["content"])
    assert untrusted == {
        "untrusted_dialogue": "USER [2026-07-21T00:00:00Z]\nhello"
    }
    assert payload["tool_choice"]["function"]["name"] == "record_distillation"
    function = payload["tools"][0]["function"]
    assert function["strict"] is True
    assert set(function["parameters"]["properties"]) == set(DISTILLATION_FIELDS)
    assert "fixture-secret-key" not in json.dumps(payload)


def test_deepseek_rejects_missing_key_and_factory_selects_provider(
    settings_factory, monkeypatch
):
    settings = _deepseek_settings(settings_factory)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    distiller = create_distiller(settings)
    assert isinstance(distiller, DeepSeekDistiller)
    with pytest.raises(DistillationUnavailable, match="DEEPSEEK_API_KEY"):
        distiller.distill("redacted dialogue")


def test_deepseek_retries_an_unstructured_response(settings_factory, monkeypatch):
    settings = _deepseek_settings(settings_factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret-key")
    structured = {
        "user_goal": "Finish the rollout",
        "summary": "The rollout was completed.",
        "outcome": "Complete",
        "decisions": [],
        "artifacts": [],
        "systems": [],
        "open_questions": [],
        "keywords": ["rollout"],
    }
    responses = iter(
        [
            _Response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "not structured", "tool_calls": []},
                        }
                    ]
                }
            ),
            _Response(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "record_distillation",
                                            "arguments": json.dumps(structured),
                                        }
                                    }
                                ]
                            },
                        }
                    ]
                }
            ),
        ]
    )
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr("distillation.urlopen", fake_urlopen)
    assert DeepSeekDistiller(settings).distill("redacted dialogue") == structured
    assert calls == 2


def test_deepseek_uses_schema_validated_json_fallback_after_strict_failures(
    settings_factory, monkeypatch
):
    settings = _deepseek_settings(settings_factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret-key")
    structured = {
        "user_goal": "Finish the rollout",
        "summary": "The rollout was completed.",
        "outcome": "Complete",
        "decisions": [],
        "artifacts": [],
        "systems": [],
        "open_questions": [],
        "keywords": ["rollout"],
    }
    bad = _Response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "invented_tool", "arguments": "{}"}}
                        ]
                    },
                }
            ]
        }
    )
    responses = iter(
        [
            bad,
            bad,
            _Response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(structured)},
                        }
                    ]
                }
            ),
        ]
    )
    payloads = []

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data.decode("utf-8")))
        return next(responses)

    monkeypatch.setattr("distillation.urlopen", fake_urlopen)
    assert DeepSeekDistiller(settings).distill("redacted dialogue") == structured
    assert len(payloads) == 3
    assert "tools" in payloads[0]
    assert payloads[2]["response_format"] == {"type": "json_object"}
    assert "tools" not in payloads[2]


def test_deepseek_sanitizes_a_no_response_claim_when_assistant_messages_exist(
    settings_factory, monkeypatch
):
    settings = _deepseek_settings(settings_factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret-key")
    invalid = {
        "user_goal": "Set up the system",
        "summary": "The dialogue contains only the user's initial question.",
        "outcome": "No assistant response was provided.",
        "decisions": [],
        "artifacts": [],
        "systems": [],
        "open_questions": [],
        "keywords": ["setup"],
    }
    response = _Response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "record_distillation",
                                    "arguments": json.dumps(invalid),
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr("distillation.urlopen", fake_urlopen)
    dialogue = "USER [2026-07-21T00:00:00Z]\nhelp\n\nASSISTANT [2026-07-21T00:01:00Z]\nworking"
    result = DeepSeekDistiller(settings).distill(dialogue)
    assert result["summary"] == ""
    assert result["outcome"] is None
    assert result["user_goal"] == invalid["user_goal"]
    assert calls == 1


def test_deepseek_accepts_only_a_whole_schema_valid_json_content(
    settings_factory, monkeypatch
):
    settings = _deepseek_settings(settings_factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret-key")
    structured = {
        "user_goal": "Finish the rollout",
        "summary": "The rollout was completed.",
        "outcome": "Complete",
        "decisions": [],
        "artifacts": [],
        "systems": [],
        "open_questions": [],
        "keywords": ["rollout"],
    }

    monkeypatch.setattr(
        "distillation.urlopen",
        lambda request, timeout: _Response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(structured)},
                    }
                ]
            }
        ),
    )
    assert DeepSeekDistiller(settings).distill("redacted dialogue") == structured


def test_deepseek_safely_repairs_unescaped_windows_path_backslashes(
    settings_factory, monkeypatch
):
    settings = _deepseek_settings(settings_factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret-key")
    structured = {
        "user_goal": "Find the report",
        "summary": "The report path was recorded.",
        "outcome": "",
        "decisions": [],
        "artifacts": [r"C:\repo\report.md"],
        "systems": [],
        "open_questions": [],
        "keywords": ["report"],
    }
    malformed = json.dumps(structured).replace(
        r"C:\\repo\\report.md",
        r"C:\repo\report.md",
    )
    monkeypatch.setattr(
        "distillation.urlopen",
        lambda request, timeout: _Response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": malformed},
                    }
                ]
            }
        ),
    )

    result = DeepSeekDistiller(settings).distill("redacted dialogue")

    assert result["artifacts"] == [r"C:\repo\report.md"]


def test_deepseek_merges_multiple_individually_valid_tool_calls(
    settings_factory, monkeypatch
):
    settings = _deepseek_settings(settings_factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret-key")
    first = {
        "user_goal": "Finish the rollout",
        "summary": "Deployment was completed.",
        "outcome": "Complete",
        "decisions": ["Keep raw citations"],
        "artifacts": [],
        "systems": ["Cerebras Memory"],
        "open_questions": [],
        "keywords": ["rollout"],
    }
    second = {
        "user_goal": "Verify the rollout",
        "summary": "Verification passed.",
        "outcome": "Complete",
        "decisions": ["Keep raw citations"],
        "artifacts": ["report.md"],
        "systems": ["Cerebras Memory"],
        "open_questions": [],
        "keywords": ["verification"],
    }
    tool_calls = [
        {"function": {"name": "untrusted_dialogue_tool", "arguments": "{}"}}
    ] + [
        {
            "function": {
                "name": "record_distillation",
                "arguments": json.dumps(record),
            }
        }
        for record in (first, second)
    ]
    monkeypatch.setattr(
        "distillation.urlopen",
        lambda request, timeout: _Response(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"tool_calls": tool_calls},
                    }
                ]
            }
        ),
    )

    result = DeepSeekDistiller(settings).distill("redacted dialogue")

    assert result["user_goal"] == "Finish the rollout; Verify the rollout"
    assert result["summary"] == "Deployment was completed.; Verification passed."
    assert result["outcome"] == "Complete"
    assert result["decisions"] == ["Keep raw citations"]
    assert result["keywords"] == ["rollout", "verification"]


def test_local_validation_enforces_storage_bounds():
    value = {field: ["x" * 500] * 20 for field in DISTILLATION_FIELDS}
    value.update({"user_goal": "g" * 700, "summary": "s" * 1_500, "outcome": "o" * 700})
    result = validate_distillation(value)
    assert len(result["user_goal"]) == 500
    assert len(result["summary"]) == 1_200
    assert len(result["outcome"]) == 500
    assert all(len(result[field]) == 12 for field in DISTILLATION_FIELDS[3:])
    assert all(len(item) <= 300 for field in DISTILLATION_FIELDS[3:7] for item in result[field])
    assert all(len(item) <= 100 for item in result["keywords"])


def test_mapped_raw_evidence_sanitizes_contradictory_no_response_claims():
    value = {
        "user_goal": "Set up Unreal MCP",
        "summary": "The dialogue contains only the user's initial question.",
        "outcome": "No assistant response was provided.",
        "decisions": [],
        "artifacts": [],
        "systems": ["Unreal Engine"],
        "open_questions": [],
        "keywords": ["MCP"],
    }

    result = sanitize_role_contradictions(
        value,
        "USER [2026-07-21T00:00:00Z]\nhelp\n\nASSISTANT [2026-07-21T00:01:00Z]\nworking",
    )

    assert result["summary"] == ""
    assert result["outcome"] is None
    assert result["user_goal"] == "Set up Unreal MCP"
