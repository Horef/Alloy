"""Small nonsecret identities for model-derived artifacts and evaluation records."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).parent


def code_fingerprint(*names: str | Path) -> str:
    """Identity of the code in the named package modules, ignoring comments, docstrings, and layout.

    Cache keys and resume signatures must change when behavior changes, but a comment or docstring
    edit must not force paid re-extraction or break resume. Hashing the parsed syntax tree (without
    docstrings or positions) gives exactly that. The tree format belongs to the Python minor version,
    so these fingerprints, and the cache entries keyed on them, are not shared across versions.
    """
    digest = hashlib.sha256()
    for name in names:
        path = name if isinstance(name, Path) else _ROOT / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (
                isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
        digest.update(path.name.encode("utf-8"))
        digest.update(ast.dump(tree, include_attributes=False).encode("utf-8"))
    return digest.hexdigest()


def records_hash(records) -> str:
    return hashlib.sha256(json.dumps([r.model_dump(mode="json") for r in records],
                                    ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_identity(settings) -> dict:
    from .llm import STRUCTURED_CALL_CONTRACT

    transport = getattr(settings, "gemini_transport", "direct")
    endpoint = getattr(settings, "apigee_base_url", "").rstrip("/") if transport == "apigee" else "direct"
    # Schemas, input normalization, and error categories decide what a judged record means; the call
    # mechanics in llm.py are versioned by STRUCTURED_CALL_CONTRACT instead of by their source.
    return {"contract_version": 3, "transport": transport,
            "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
            "structured_call_contract": STRUCTURED_CALL_CONTRACT,
            "supporting_implementation_sha256": code_fingerprint("models.py", "validation.py", "response_errors.py")}


def cache_identity(settings) -> dict:
    """Model-route identity for corpus caches, whose stage keys already fingerprint prompts and schemas.

    Unlike :func:`model_identity` it does not hash whole modules, so a logging or retry edit in
    ``llm.py`` does not force re-extracting every document.
    """
    from .llm import STRUCTURED_CALL_CONTRACT

    transport = getattr(settings, "gemini_transport", "direct")
    endpoint = getattr(settings, "apigee_base_url", "").rstrip("/") if transport == "apigee" else "direct"
    return {"contract_version": 3, "transport": transport,
            "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
            "structured_call_contract": STRUCTURED_CALL_CONTRACT,
            "seed": getattr(settings, "seed", None)}
