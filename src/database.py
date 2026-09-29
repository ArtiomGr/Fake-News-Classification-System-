"""JSON persistence for anonymous analysis history."""
import json
import hashlib
import os
from threading import RLock
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "analysis_history.json"
_HISTORY_LOCK = RLock()


def initialize_database(database_path=DEFAULT_HISTORY_PATH):
	"""Create an empty JSON history file when it does not exist yet."""
	database_path = Path(database_path)
	database_path.parent.mkdir(parents=True, exist_ok=True)
	if not database_path.exists():
		database_path.write_text("[]\n", encoding="utf-8")


def _read_history(database_path):
	try:
		value = json.loads(Path(database_path).read_text(encoding="utf-8"))
	except json.JSONDecodeError as error:
		raise ValueError(f"History file is not valid JSON: {database_path}") from error
	if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
		raise ValueError(f"History file must contain a JSON array of records: {database_path}")
	return value


def _write_history(database_path, history):
	database_path = Path(database_path)
	temporary_path = database_path.with_suffix(database_path.suffix + ".tmp")
	temporary_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	os.replace(temporary_path, database_path)


def save_analysis(text, result, database_path=DEFAULT_HISTORY_PATH):
	"""Store result metadata without retaining the submitted text."""
	if not text or not isinstance(result, dict):
		raise ValueError("Text and prediction result are required")
	initialize_database(database_path)
	text_value = str(text)
	record = {
		"id": None,
		"analyzed_at": datetime.now(timezone.utc).isoformat(),
		"text_sha256": hashlib.sha256(text_value.encode("utf-8")).hexdigest(),
		"text_length": len(text_value),
		"system": result["system"],
		"classifier": result["classifier"],
		"predicted_label": result["predicted_label"],
		"confidence": float(result["confidence"]),
		"number_of_chunks": int(result["number_of_chunks"]),
		"inference_seconds": float(result["inference_seconds"]),
	}
	with _HISTORY_LOCK:
		history = _read_history(database_path)
		record["id"] = max((int(item.get("id", 0)) for item in history), default=0) + 1
		history.append(record)
		_write_history(database_path, history)


def get_recent_analyses(limit=20, database_path=DEFAULT_HISTORY_PATH):
	"""Return recent anonymous analyses, newest first."""
	if not isinstance(limit, int) or limit <= 0:
		raise ValueError("History limit must be a positive integer")
	initialize_database(database_path)
	with _HISTORY_LOCK:
		history = _read_history(database_path)
	return list(reversed(history[-limit:]))
