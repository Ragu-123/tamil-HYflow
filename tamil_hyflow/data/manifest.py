from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

@dataclass
class ManifestRecord:
    audio: str
    text: str
    speaker_id: str = "unknown"
    language: str = "ta"
    sample_rate: int | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ManifestRecord":
        return cls(**item)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def read_manifest(path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(ManifestRecord.from_dict(json.loads(line)))
    return records

def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
