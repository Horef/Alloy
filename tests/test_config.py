import pytest

from chatbot_eval.config import load_settings


def test_configuration_rejects_invalid_cross_field_values(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[generation]
batch_chunks = 0
unanswerable_ratio = 0.6
user_variation_ratio = 0.5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_chunks.*unanswerable_ratio"):
        load_settings(path, require_api_key=False)


def test_apigee_transport_uses_its_own_environment_key(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[gemini]
transport = "apigee"
apigee_api_key_env = "TEST_APIGEE_KEY"
apigee_base_url = "https://preprod.apigee.digital.idf.il/ai_gateway/v1/hr"
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("TEST_APIGEE_KEY", "apigee-secret")

    settings = load_settings(path)

    assert settings.gemini_transport == "apigee"
    assert settings.api_key == ""
    assert settings.apigee_api_key == "apigee-secret"
    assert settings.apigee_base_url.endswith("/ai_gateway/v1/hr")


def test_apigee_transport_requires_https_ai_gateway_url(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[gemini]
transport = "apigee"
apigee_base_url = "http://example.test/not-the-gateway"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="HTTPS AI Gateway"):
        load_settings(path, require_api_key=False)


def test_cache_defaults_are_enabled_and_local(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("", encoding="utf-8")

    settings = load_settings(path, require_api_key=False)

    assert settings.cache_enabled is True
    assert settings.cache_directory == ".chatbot_eval_cache"


def test_configuration_rejects_empty_cache_directory(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[cache]\ndirectory = "  "\n', encoding="utf-8")

    with pytest.raises(ValueError, match="cache.directory"):
        load_settings(path, require_api_key=False)


def test_evaluation_safety_limits_have_conservative_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("", encoding="utf-8")

    settings = load_settings(path, require_api_key=False)

    assert settings.judge_max_answer_chars == 20_000
    assert settings.judge_max_context_chars == 60_000
    assert settings.insights_max_prompt_chars == 80_000
    assert settings.gemini_request_timeout_seconds == 120
    assert settings.max_concurrency == 1
    assert settings.prompt_max_document_chars == 100_000
    assert settings.prompt_max_evaluation_chars == 60_000
    assert settings.prompt_max_auxiliary_chars == 30_000
    assert settings.prompt_instruction_profile == "guided"
    assert settings.prompt_answer_policy == "balanced"


def test_prompt_profile_and_answer_policy_are_configurable(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[generation]
prompt_instruction_profile = "compact"
prompt_answer_policy = "conservative"
""",
        encoding="utf-8",
    )

    settings = load_settings(path, require_api_key=False)

    assert settings.prompt_instruction_profile == "compact"
    assert settings.prompt_answer_policy == "conservative"


def test_configuration_rejects_invalid_evaluation_safety_limits(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[evaluation]
judge_max_answer_chars = 10
insights_max_prompt_chars = 10
gemini_request_timeout_seconds = 0
max_concurrency = 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="character limits.*insights.*timeout.*concurrency"):
        load_settings(path, require_api_key=False)
