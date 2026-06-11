from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


SMOKE_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "generation_config.json",
]

WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.json",
    "tokenizer*",
    "*.model",
    "*.txt",
    "*.py",
]


@dataclass(frozen=True)
class ModelCheck:
    name: str
    model_id: str | None
    status: str
    detail: str
    gated: str = "unknown"
    private: str = "unknown"
    cached_path: str = ""


def load_model_specs(models_path: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
    specs = list(config.get("zero_shot_llms", []))
    for task_models in config.get("pretrained_or_finetuned_task_models", {}).values():
        specs.extend(task_models or [])
    return specs


def token_from_env(token_env: str, anonymous: bool) -> str | None:
    if anonymous:
        return None
    token = os.getenv(token_env)
    return token.strip() if token and token.strip() else None


def short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    return message[:240] + ("..." if len(message) > 240 else "")


def classify_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, GatedRepoError):
        return (
            "auth_required",
            "gated repo: accept the model license on Hugging Face and use an HF token",
        )
    if isinstance(exc, RepositoryNotFoundError):
        return (
            "not_found_or_private",
            "repo not found, private, renamed, or inaccessible with current token",
        )
    if isinstance(exc, RevisionNotFoundError):
        return ("bad_revision", "revision not found")
    if isinstance(exc, HfHubHTTPError):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code in {401, 403}:
            return (
                "auth_required",
                "authentication or license acceptance required",
            )
        if code == 404:
            return (
                "not_found_or_private",
                "repo/file not found, private, renamed, or inaccessible",
            )
        return ("hub_error", f"Hugging Face HTTP error {code or 'unknown'}")
    return ("error", exc.__class__.__name__)


def smoke_download_files(model_id: str, token: str | None, cache_dir: Path | None) -> str:
    downloaded = []
    last_error: Exception | None = None
    for filename in SMOKE_FILES:
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                token=token,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            downloaded.append(filename)
            if filename == "config.json":
                break
        except HfHubHTTPError as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                raise
        except Exception as exc:
            last_error = exc
    if downloaded:
        return ", ".join(downloaded)
    if last_error:
        raise last_error
    return "no smoke files attempted"


def check_model(
    api: HfApi,
    spec: dict[str, Any],
    token: str | None,
    cache_dir: Path | None,
    download_weights: bool,
) -> ModelCheck:
    name = str(spec.get("name") or "")
    model_id = spec.get("model_id")
    if not model_id:
        return ModelCheck(
            name=name,
            model_id=None,
            status="skipped",
            detail="model_id is null placeholder",
            gated="n/a",
            private="n/a",
        )

    try:
        info = api.model_info(str(model_id), token=token)
        gated = str(getattr(info, "gated", "unknown"))
        private = str(getattr(info, "private", "unknown"))
        files = smoke_download_files(str(model_id), token=token, cache_dir=cache_dir)
        cached_path = ""
        detail = f"accessible; smoke file(s): {files}"

        if download_weights:
            cached_path = snapshot_download(
                repo_id=str(model_id),
                token=token,
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=WEIGHT_PATTERNS,
            )
            detail = "full snapshot cached"

        return ModelCheck(
            name=name,
            model_id=str(model_id),
            status="ok",
            detail=detail,
            gated=gated,
            private=private,
            cached_path=cached_path,
        )
    except Exception as exc:
        status, detail = classify_error(exc)
        return ModelCheck(
            name=name,
            model_id=str(model_id),
            status=status,
            detail=f"{detail}; {short_error(exc)}",
        )


def print_table(checks: list[ModelCheck]) -> None:
    headers = ["status", "name", "model_id", "gated", "detail"]
    rows = [
        [
            check.status,
            check.name,
            check.model_id or "",
            check.gated,
            check.detail,
        ]
        for check in checks
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def print_colab_help(token_env: str) -> None:
    print(
        f"""

Colab setup suggestion:

from google.colab import drive, userdata
drive.mount('/content/drive')

import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/content/drive/MyDrive/hf_cache/transformers'
os.environ['{token_env}'] = userdata.get('{token_env}') or ''

# Put your Hugging Face token in Colab Secrets as {token_env}.
# For gated models, also open the model page on Hugging Face and accept the license.

!python -m src.check_hf_models --models configs/models.yaml --cache-dir "$HF_HOME"

# Optional: intentionally pre-download one model snapshot into Drive cache.
# !python -m src.check_hf_models --models configs/models.yaml --cache-dir "$HF_HOME" --only "Qwen 2.5 3B Instruct" --download-weights
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-check Hugging Face model access and optional Drive-backed caching."
    )
    parser.add_argument("--models", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="Ignore any token and test whether models work without authentication.",
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download full model snapshots into cache. This can be very large.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Check only model names or model_ids matching this value. Can be repeated.",
    )
    parser.add_argument("--print-colab-help", action="store_true")
    args = parser.parse_args()

    token = token_from_env(args.token_env, anonymous=args.anonymous)
    specs = load_model_specs(args.models)
    if args.only:
        requested = set(args.only)
        specs = [
            spec
            for spec in specs
            if spec.get("name") in requested or spec.get("model_id") in requested
        ]
    if not specs:
        raise SystemExit("No model specs matched.")

    api = HfApi()
    checks = [
        check_model(
            api=api,
            spec=spec,
            token=token,
            cache_dir=args.cache_dir,
            download_weights=args.download_weights,
        )
        for spec in specs
    ]
    print_table(checks)

    blocked = [check for check in checks if check.status in {"auth_required", "not_found_or_private"}]
    if blocked:
        print(
            "\nSome models need authentication/license acceptance or are not publicly accessible. "
            f"Set {args.token_env}, accept gated model licenses on Hugging Face, then rerun without --anonymous."
        )
    if args.print_colab_help:
        print_colab_help(args.token_env)


if __name__ == "__main__":
    main()
