"""
Debug-trace logging: one JSONL file per stage (not one .txt per item).

Why: per-item .txt produced thousands of tiny files (e.g. sc_mcp_logs =
#items x #cases) → painful to zip/copy. JSONL keeps it to ONE file per stage,
still greppable for a single item (`grep '"idx": 5'` / jq) and still readable by
Claude one record at a time.

Toggle with env EDC_DEBUG_LOGS (default ON):
  EDC_DEBUG_LOGS=0  -> skip all debug traces (production runs)
"""
import os
import json


def debug_logs_enabled() -> bool:
    return os.environ.get("EDC_DEBUG_LOGS", "1").strip().lower() not in {"0", "false", "no", "off"}


def jsonl_log(iter_dir: str, stage: str, record: dict) -> None:
    """Append one JSON line to <iter_dir>/<stage>.jsonl (no-op if disabled)."""
    if not iter_dir or not debug_logs_enabled():
        return
    try:
        os.makedirs(iter_dir, exist_ok=True)
        with open(os.path.join(iter_dir, f"{stage}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
