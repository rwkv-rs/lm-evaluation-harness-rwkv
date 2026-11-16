import hashlib
import os
from pathlib import Path
from typing import Any

import datasets


DATA_ROOT_ENV = "MODEL_WRITTEN_EVALS_DATA_ROOT"


def local_source_path(url: str, root: Path) -> Path:
    suffix = Path(url).suffix or ".jsonl"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return root / f"{digest}{suffix}"


def _resolve_data_files(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_data_files(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_data_files(item, root) for item in value]
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        path = local_source_path(value, root)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing materialized model-written source: {path}"
            )
        return str(path)
    return value


def load_json_dataset(data_files, **kwargs):
    data_root = os.environ.get(DATA_ROOT_ENV)
    resolved = (
        _resolve_data_files(data_files, Path(data_root)) if data_root else data_files
    )
    if data_root:
        return datasets.load_dataset("json", data_files=resolved)
    # Remote http(s) URLs go through fsspec's aiohttp-backed HTTPFileSystem,
    # which ignores HTTP_PROXY/HTTPS_PROXY/ALL_PROXY unless trust_env is set
    # explicitly (aiohttp.ClientSession defaults trust_env=False).
    return datasets.load_dataset(
        "json",
        data_files=resolved,
        storage_options={"client_kwargs": {"trust_env": True}},
    )
