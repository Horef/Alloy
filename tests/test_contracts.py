from chatbot_eval.contracts import code_fingerprint


def test_code_fingerprint_ignores_comments_docstrings_and_layout(tmp_path):
    module = tmp_path / "stage.py"

    def fingerprint(source):
        module.write_text(source, encoding="utf-8")
        return code_fingerprint(module)

    base = fingerprint('PROMPT = "extract"\n\ndef run(x):\n    return x + 1\n')
    assert fingerprint(
        '"""Module docs."""\n# a comment\nPROMPT = "extract"\n\n\ndef run(x):\n    """Docs."""\n    return x + 1  # why\n'
    ) == base
    # Behavior and prompt text are part of the identity.
    assert fingerprint('PROMPT = "extract"\n\ndef run(x):\n    return x + 2\n') != base
    assert fingerprint('PROMPT = "extract all"\n\ndef run(x):\n    return x + 1\n') != base
