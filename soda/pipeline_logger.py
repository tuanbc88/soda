import os
import json
import time
from typing import Any, Dict, List


class PipelineLogger:
    """
    Append-only logger: each event is written as a JSON line (NDJSON) on flush,
    so cost per log() is O(1) instead of O(n) for the full rewrite.
    Read back with: [json.loads(l) for l in open(path) if l.strip()]
    """

    def __init__(self, save_path: str, auto_flush: bool = True):
        self.save_path = save_path
        self.auto_flush = auto_flush

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Truncate/create the file so each run starts clean.
        open(self.save_path, "w", encoding="utf-8").close()

    def log(
        self,
        item_id: int,
        stage: str,
        data: Dict[str, Any],
        event_type: str = "result",
        meta: Dict[str, Any] = None,
    ):
        event = {
            "timestamp": time.time(),
            "item_id": item_id,
            "stage": stage,       # oie / sd1 / sd2 / sd3 / sc / final
            "type": event_type,   # result / error / debug
            "data": data,
        }

        if meta:
            event["meta"] = meta

        # Append one line and retain nothing in memory (avoids RAM growth on
        # large datasets — a contributor to the SD OOM-kill).
        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_error(self, item_id: int, stage: str, error: Exception):
        self.log(
            item_id=item_id,
            stage=stage,
            event_type="error",
            data={
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )