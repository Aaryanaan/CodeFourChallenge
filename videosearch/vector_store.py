"""LanceDB vector store for chunk embeddings (IDX-02, IDX-04)."""

from pathlib import Path

import lancedb
import pyarrow as pa

_DEFAULT_VECTOR_DIM = 768


def _chunks_schema(dim: int = _DEFAULT_VECTOR_DIM) -> pa.Schema:
    """Build LanceDB schema with the given vector dimensionality."""
    return pa.schema([
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("video_id", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("start_time", pa.float64()),
        pa.field("end_time", pa.float64()),
        pa.field("duration", pa.float64()),
        pa.field("combined_text", pa.string()),
        pa.field("visual_caption", pa.string()),  # stored separately for display
        pa.field("volume_level", pa.string()),    # "quiet" | "normal" | "loud" (D-05)
        pa.field("has_speech", pa.bool_()),       # (D-06)
        pa.field("has_ocr", pa.bool_()),          # (D-06)
        pa.field("has_raised_voice", pa.bool_()), # (D-06)
        pa.field("scene_type", pa.string()),
    ])


class LanceVectorStore:
    """Stores and queries chunk embeddings in LanceDB. Implements VectorStore protocol.

    Single 'chunks' table per D-11. Denormalized schema per D-12.
    Upsert by compound key (video_id, chunk_index) per D-13.
    """

    def __init__(self, index_dir: str | Path = "data/index", vector_dim: int = _DEFAULT_VECTOR_DIM):
        self._index_dir = Path(index_dir)
        self._lancedb_dir = self._index_dir / "lancedb"
        self._lancedb_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._lancedb_dir))
        self._vector_dim = vector_dim
        self._schema = _chunks_schema(vector_dim)
        self._table = None

    def _get_table(self):
        if self._table is None:
            if "chunks" in self._db.table_names():
                # Open existing table (may have different vector dim than config)
                self._table = self._db.open_table("chunks")
            else:
                self._table = self._db.create_table(
                    "chunks", schema=self._schema, exist_ok=True
                )
        return self._table

    def clear(self) -> None:
        """Drop and recreate the chunks table, applying the current schema."""
        if "chunks" in self._db.table_names():
            self._db.drop_table("chunks")
        self._table = None

    def upsert(self, rows: list[dict]) -> None:
        """Upsert rows by compound key (video_id, chunk_index) per D-04/D-13."""
        table = self._get_table()
        (
            table
            .merge_insert(["video_id", "chunk_index"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def count(self) -> int:
        """Return the number of rows in the vector store."""
        return self._get_table().count_rows()

    def stored_vector_dim(self) -> int | None:
        """Return the vector dimension of the stored table, or None if empty."""
        if "chunks" not in self._db.table_names():
            return None
        table = self._get_table()
        schema = table.schema
        for field in schema:
            if field.name == "vector":
                return field.type.list_size
        return None

    def count_by_video(self, video_id: str) -> int:
        """Count existing rows for a video_id. Used for incremental skip detection (IDX-05)."""
        table = self._get_table()
        return table.count_rows(filter=f"video_id = '{video_id}'")

    def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Vector similarity search with optional metadata filter (IDX-04).

        Returns list of dicts with all schema columns plus '_distance'.
        Lower _distance = more similar (cosine distance).
        """
        table = self._get_table()
        query = table.search(vector).distance_type("cosine").limit(top_k)
        if filter_expr:
            query = query.where(filter_expr)
        return query.to_list()
