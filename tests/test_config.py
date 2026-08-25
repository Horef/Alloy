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
