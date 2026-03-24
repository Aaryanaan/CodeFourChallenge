"""Metadata writer for JSON serialization of ChunkMetadata.

Writes and loads per-video ChunkMetadata as JSON files at
data/metadata/{video_id}.json. Per D-11: one JSON file per video
containing a list of all ChunkMetadata for that video.
"""

import json
from pathlib import Path

from videosearch.models import ChunkMetadata


class MetadataWriter:
    """Write and load per-video ChunkMetadata as JSON files.

    Per D-11: one JSON file per video at data/metadata/{video_id}.json
    containing a list of all ChunkMetadata for that video.
    """

    def __init__(self, metadata_dir: str | Path = "data/metadata"):
        self.metadata_dir = Path(metadata_dir)

    def write(self, video_id: str, chunks: list[ChunkMetadata]) -> Path:
        """Serialize list of ChunkMetadata to JSON file.

        Creates metadata_dir if it does not exist.
        Returns path to written file.

        Raises ValueError if any chunk's video_id does not match the
        given video_id, preventing silent cross-video contamination.
        """
        mismatched = [
            c.chunk_index for c in chunks if c.video_id != video_id
        ]
        if mismatched:
            raise ValueError(
                f"Chunk(s) {mismatched} have video_id mismatching "
                f"target '{video_id}'"
            )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.metadata_dir / f"{video_id}.json"
        data = [chunk.model_dump(mode="json") for chunk in chunks]
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return output_path

    def load(self, video_id: str) -> list[ChunkMetadata]:
        """Load list of ChunkMetadata from JSON file.

        Raises FileNotFoundError if file does not exist.
        """
        file_path = self.metadata_dir / f"{video_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")
        data = json.loads(file_path.read_text())
        return [ChunkMetadata.model_validate(item) for item in data]
