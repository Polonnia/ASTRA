import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


_STAGE_CTX: ContextVar[str] = ContextVar("token_usage_stage", default="unspecified")
_LOCK = threading.Lock()
_STATS: Dict[str, Dict[str, Any]] = {}


def current_stage() -> str:
    return _STAGE_CTX.get()


@contextmanager
def token_usage_stage(stage_name: str):
    normalized = str(stage_name or "").strip() or "unspecified"
    token = _STAGE_CTX.set(normalized)
    try:
        yield
    finally:
        _STAGE_CTX.reset(token)


def _ensure_stage(stage_name: str) -> Dict[str, Any]:
    if stage_name not in _STATS:
        _STATS[stage_name] = {
            "api_calls": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "models": {},
            "sources": {},
        }
    return _STATS[stage_name]


def _bump_counter(container: Dict[str, int], key: str, value: int) -> None:
    if value <= 0:
        return
    container[key] = int(container.get(key, 0)) + int(value)


def record_chat_usage(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    model: str = "",
    stage: str | None = None,
    source: str = "",
    cache_hit: bool = False,
    api_call: bool = True,
) -> None:
    stage_name = str(stage or current_stage() or "unspecified").strip() or "unspecified"
    model_name = str(model or "unknown").strip() or "unknown"
    source_name = str(source or "unknown").strip() or "unknown"

    with _LOCK:
        stage_bucket = _ensure_stage(stage_name)
        if api_call:
            stage_bucket["api_calls"] += 1
        if cache_hit:
            stage_bucket["cache_hits"] += 1

        stage_bucket["prompt_tokens"] += max(int(prompt_tokens or 0), 0)
        stage_bucket["completion_tokens"] += max(int(completion_tokens or 0), 0)

        final_total = int(total_tokens or 0)
        if final_total <= 0:
            final_total = max(int(prompt_tokens or 0), 0) + max(int(completion_tokens or 0), 0)
        stage_bucket["total_tokens"] += max(final_total, 0)

        model_bucket = stage_bucket["models"].setdefault(
            model_name,
            {
                "api_calls": 0,
                "cache_hits": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        if api_call:
            model_bucket["api_calls"] += 1
        if cache_hit:
            model_bucket["cache_hits"] += 1
        model_bucket["prompt_tokens"] += max(int(prompt_tokens or 0), 0)
        model_bucket["completion_tokens"] += max(int(completion_tokens or 0), 0)
        model_bucket["total_tokens"] += max(final_total, 0)

        _bump_counter(stage_bucket["sources"], source_name, 1)


def reset_token_usage_stats() -> None:
    with _LOCK:
        _STATS.clear()


def get_token_usage_stats() -> Dict[str, Any]:
    with _LOCK:
        per_stage = {k: json.loads(json.dumps(v)) for k, v in _STATS.items()}

    summary = {
        "api_calls": 0,
        "cache_hits": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for stage_value in per_stage.values():
        summary["api_calls"] += int(stage_value.get("api_calls", 0))
        summary["cache_hits"] += int(stage_value.get("cache_hits", 0))
        summary["prompt_tokens"] += int(stage_value.get("prompt_tokens", 0))
        summary["completion_tokens"] += int(stage_value.get("completion_tokens", 0))
        summary["total_tokens"] += int(stage_value.get("total_tokens", 0))

    ordered_stages = sorted(
        per_stage.items(), key=lambda x: int(x[1].get("total_tokens", 0)), reverse=True
    )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": summary,
        "stages": {k: v for k, v in ordered_stages},
    }


def save_token_usage_stats(output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = get_token_usage_stats()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def format_token_usage_report() -> str:
    payload = get_token_usage_stats()
    summary = payload.get("summary", {})
    lines = [
        "========== Token Usage Report ==========" ,
        (
            "Total: api_calls={api_calls}, cache_hits={cache_hits}, prompt_tokens={prompt_tokens}, "
            "completion_tokens={completion_tokens}, total_tokens={total_tokens}"
        ).format(
            api_calls=summary.get("api_calls", 0),
            cache_hits=summary.get("cache_hits", 0),
            prompt_tokens=summary.get("prompt_tokens", 0),
            completion_tokens=summary.get("completion_tokens", 0),
            total_tokens=summary.get("total_tokens", 0),
        ),
        "",
    ]

    for stage_name, stage_data in payload.get("stages", {}).items():
        lines.append(
            (
                "[{stage}] api_calls={api_calls}, cache_hits={cache_hits}, prompt_tokens={prompt_tokens}, "
                "completion_tokens={completion_tokens}, total_tokens={total_tokens}"
            ).format(
                stage=stage_name,
                api_calls=stage_data.get("api_calls", 0),
                cache_hits=stage_data.get("cache_hits", 0),
                prompt_tokens=stage_data.get("prompt_tokens", 0),
                completion_tokens=stage_data.get("completion_tokens", 0),
                total_tokens=stage_data.get("total_tokens", 0),
            )
        )

    if len(lines) == 3:
        lines.append("No usage has been recorded.")

    return "\n".join(lines)
