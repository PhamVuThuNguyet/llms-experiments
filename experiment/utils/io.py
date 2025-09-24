import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str | Path) -> Any:
	p = Path(path)
	with p.open('r', encoding='utf-8') as f:
		return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
	p = Path(path)
	with p.open('r', encoding='utf-8') as f:
		return json.load(f)


def ensure_dir(path: str | Path) -> Path:
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def write_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
	p = Path(path)
	p.parent.mkdir(parents=True, exist_ok=True)
	with p.open('a', encoding='utf-8') as f:
		for rec in records:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")
