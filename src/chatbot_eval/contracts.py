"""Small nonsecret identities for model-derived artifacts and evaluation records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def records_hash(records) -> str:
    return hashlib.sha256(json.dumps([r.model_dump(mode="json") for r in records],
                                    ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_identity(settings) -> dict:
    transport = getattr(settings, "gemini_transport", "direct")
    endpoint = getattr(settings, "apigee_base_url", "").rstrip("/") if transport == "apigee" else "direct"
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("models.py", "validation.py", "llm.py", "response_errors.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return {"contract_version": 2, "transport": transport,
            "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
            "supporting_implementation_sha256": digest.hexdigest()}


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
