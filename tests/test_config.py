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
