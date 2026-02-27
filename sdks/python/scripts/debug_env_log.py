import json
import os
import time


def _write(payload: dict) -> None:
  log_path = os.path.join(os.getcwd(), "debug-0b9657.log")
  with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")


_write({
    "sessionId": "0b9657",
    "timestamp": int(time.time() * 1000),
    "location": "sdks/python/scripts/debug_env_log.py:1",
    "message": "tox env snapshot",
    "runId": os.environ.get("TOX_ENV_NAME", ""),
    "hypothesisId": "H1",
    "data": {
        "TOX_ENV_NAME": os.environ.get("TOX_ENV_NAME", ""),
        "DEPS": os.environ.get("DEPS", ""),
        "COMMON_DEPS": os.environ.get("COMMON_DEPS", ""),
        "PIP_PRE": os.environ.get("PIP_PRE", ""),
        "VIRTUAL_ENV": bool(os.environ.get("VIRTUAL_ENV")),
    },
})

