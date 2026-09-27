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


def test_answer_completeness_verification_defaults_off_and_is_configurable(tmp_path):
    default_path = tmp_path / "default.toml"
    default_path.write_text("", encoding="utf-8")
    default_settings = load_settings(default_path, require_api_key=False)
    assert default_settings.verify_answer_completeness is False
    assert default_settings.completeness_evidence_limit == 16

    enabled_path = tmp_path / "enabled.toml"
    enabled_path.write_text(
        "[generation]\nverify_answer_completeness = true\ncompleteness_evidence_limit = 24\n",
        encoding="utf-8",
    )
    enabled_settings = load_settings(enabled_path, require_api_key=False)
    assert enabled_settings.verify_answer_completeness is True
    assert enabled_settings.completeness_evidence_limit == 24


def test_configuration_rejects_nonpositive_completeness_evidence_limit(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[generation]\ncompleteness_evidence_limit = 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completeness_evidence_limit"):
        load_settings(path, require_api_key=False)


def test_theme_layer_defaults_to_theme_mode_and_is_configurable(tmp_path):
    default_path = tmp_path / "default.toml"
    default_path.write_text("", encoding="utf-8")
    default_settings = load_settings(default_path, require_api_key=False)
    assert default_settings.topic_mode == "theme"
    assert default_settings.extract_themes is False
    assert default_settings.max_theme_vocabulary == 20

    entity_path = tmp_path / "entity.toml"
    entity_path.write_text(
        "[generation]\ntopic_mode = \"entity\"\nextract_themes = true\nmax_theme_vocabulary = 30\n",
        encoding="utf-8",
    )
    entity_settings = load_settings(entity_path, require_api_key=False)
    assert entity_settings.topic_mode == "entity"
    assert entity_settings.extract_themes is True
    assert entity_settings.max_theme_vocabulary == 30


def test_configuration_rejects_unknown_topic_mode(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[generation]\ntopic_mode = "semantic"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="topic_mode must be 'entity' or 'theme'"):
        load_settings(path, require_api_key=False)


def test_configuration_rejects_nonpositive_theme_vocabulary(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[generation]\nmax_theme_vocabulary = 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_theme_vocabulary must be positive"):
        load_settings(path, require_api_key=False)


def test_optional_generation_checks_default_off_and_are_validated(tmp_path):
    default_path = tmp_path / "default.toml"
    default_path.write_text("", encoding="utf-8")
    settings = load_settings(default_path, require_api_key=False)
    assert settings.embedding_model == ""
    assert not settings.verify_unanswerable and not settings.filter_closed_book_answerable
    assert not settings.continue_on_call_failure
    assert settings.variation_batch_size == 30 and settings.semantic_duplicate_threshold == 0.92

    bad_path = tmp_path / "bad.toml"
    bad_path.write_text("[generation]\nsemantic_duplicate_threshold = 1.5\nvariation_batch_size = 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantic_duplicate_threshold.*variation_batch_size"):
        load_settings(bad_path, require_api_key=False)
