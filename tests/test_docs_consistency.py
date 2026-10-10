"""Guards that keep the documentation in step with the code.

Each test names the file to update when it fails. They exist because settings, modules, and CLI
options were repeatedly added without updating the README, the example config, or the design doc.
"""
import re
import tomllib
from pathlib import Path

from chatbot_eval.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
ARCHITECTURE = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
# Read for backward compatibility only; documented as legacy in the README, absent from the example.
LEGACY_SETTINGS = {"generation.filter_closed_book_answerable"}


def _toml_settings() -> set[str]:
    source = (ROOT / "src" / "chatbot_eval" / "config.py").read_text(encoding="utf-8")
    return {f"{section}.{key}" for section, key in re.findall(r'_get\(\s*data,\s*"(\w+)",\s*"(\w+)"', source)}


def test_every_setting_is_in_the_readme_table_and_the_example_config():
    settings = _toml_settings()
    assert len(settings) > 50, "the settings pattern no longer matches config.py"
    example = tomllib.loads((ROOT / "config.example.toml").read_text(encoding="utf-8"))
    in_example = {f"{section}.{key}" for section, table in example.items() for key in table}
    missing_readme = sorted(name for name in settings if f"| `{name}` |" not in README)
    missing_example = sorted(settings - in_example - LEGACY_SETTINGS)
    assert not missing_readme, f"add these settings to the README parameter table: {missing_readme}"
    assert not missing_example, f"add these settings to config.example.toml: {missing_example}"
    assert not in_example - settings, f"config.example.toml has keys config.py ignores: {sorted(in_example - settings)}"


def test_every_module_is_in_the_readme_code_map_and_the_architecture_doc():
    modules = sorted(path.name for path in (ROOT / "src" / "chatbot_eval").glob("*.py") if path.name != "__init__.py")
    missing_readme = [name for name in modules if f"`{name}`" not in README]
    missing_architecture = [name for name in modules if f"`{name}`" not in ARCHITECTURE]
    assert not missing_readme, f"add these modules to the README code map: {missing_readme}"
    assert not missing_architecture, f"describe these modules in docs/architecture.md: {missing_architecture}"


def test_every_cli_command_and_option_is_documented_in_the_readme():
    parser = build_parser()
    commands = next(action for action in parser._actions if action.dest == "command").choices
    missing = []
    for name, subparser in commands.items():
        if f"`{name}`" not in README:
            missing.append(name)
        for action in subparser._actions:
            for option in action.option_strings:
                if option.startswith("--") and option != "--help" and f"`{option}" not in README:
                    missing.append(f"{name} {option}")
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith("--") and option != "--help" and f"`{option}" not in README:
                missing.append(option)
    assert not missing, f"document these commands/options in the README: {sorted(set(missing))}"


def test_relative_links_in_agent_and_design_docs_resolve():
    broken = []
    for document in ("README.md", "AGENTS.md", "docs/architecture.md"):
        path = ROOT / document
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8")):
            if not re.match(r"[a-z]+://", target) and not (path.parent / target).exists():
                broken.append(f"{document} -> {target}")
    assert not broken, f"fix these relative links: {broken}"
