"""Tests for LanceVectorStore (IDX-02, IDX-04).

All tests use tmp_path for isolated LanceDB directories.
"""

import pytest

from videosearch.protocols import VectorStore
from videosearch.vector_store import _chunks_schema, _DEFAULT_VECTOR_DIM as VECTOR_DIM, LanceVectorStore


@pytest.fixture
def store(tmp_path):
    """LanceVectorStore backed by a temporary directory."""
    return LanceVectorStore(index_dir=tmp_path)


def _make_row(video_id="vid_001", chunk_index=0, **overrides):
    """Helper to build a valid row dict."""
    row = {
        "vector": [float(chunk_index * 0.1 + 1.0 + i * 0.01) for i in range(VECTOR_DIM)],
        "video_id": video_id,
        "chunk_index": chunk_index,
        "start_time": 0.0,
        "end_time": 30.0,
        "duration": 30.0,
        "combined_text": "test text",
        "volume_level": "normal",
        "has_speech": True,
        "has_ocr": False,
        "has_raised_voice": False,
        "scene_type": "action",
    }
    row.update(overrides)
    return row


def test_table_schema(store):
    """Table schema has all required columns."""
    table = store._get_table()
    schema = table.schema
    expected_columns = [
        "vector", "video_id", "chunk_index", "volume_level",
        "has_speech", "has_ocr", "has_raised_voice", "duration",
    ]
    column_names = [f.name for f in schema]
    for col in expected_columns:
        assert col in column_names, f"Missing column: {col}"


def test_upsert_roundtrip(store):
    """Upserting same row twice yields 1 row not 2 (compound key)."""
    row = _make_row(video_id="vid_001", chunk_index=0, combined_text="first")
    store.upsert([row])

    row_updated = _make_row(video_id="vid_001", chunk_index=0, combined_text="second")
    store.upsert([row_updated])

    table = store._get_table()
    df = table.to_pandas()
    matching = df[(df["video_id"] == "vid_001") & (df["chunk_index"] == 0)]
    assert len(matching) == 1
    assert matching.iloc[0]["combined_text"] == "second"


def test_search_returns_results(store):
    """search returns dicts with _distance key."""
    rows = [_make_row(chunk_index=i) for i in range(3)]
    store.upsert(rows)

    query_vector = [1.0] * VECTOR_DIM
    results = store.search(query_vector, top_k=3)
    assert len(results) > 0
    assert "_distance" in results[0]


def test_filter_by_raised_voice(store):
    """search with filter_expr='has_raised_voice = true' only returns matching rows."""
    rows = [
        _make_row(chunk_index=0, has_raised_voice=True),
        _make_row(chunk_index=1, has_raised_voice=False),
        _make_row(chunk_index=2, has_raised_voice=True),
    ]
    store.upsert(rows)

    query_vector = [1.0] * VECTOR_DIM
    results = store.search(query_vector, top_k=10, filter_expr="has_raised_voice = true")
    assert len(results) == 2
    assert all(r["has_raised_voice"] for r in results)


def test_filter_by_duration(store):
    """search with filter_expr='duration >= 30.0' only returns matching rows."""
    rows = [
        _make_row(chunk_index=0, duration=15.0),
        _make_row(chunk_index=1, duration=30.0),
        _make_row(chunk_index=2, duration=45.0),
    ]
    store.upsert(rows)

    query_vector = [1.0] * VECTOR_DIM
    results = store.search(query_vector, top_k=10, filter_expr="duration >= 30.0")
    assert len(results) == 2
    assert all(r["duration"] >= 30.0 for r in results)


def test_search_with_no_filter(store):
    """search without filter returns all rows."""
    rows = [_make_row(chunk_index=i) for i in range(3)]
    store.upsert(rows)

    query_vector = [1.0] * VECTOR_DIM
    results = store.search(query_vector, top_k=10)
    assert len(results) == 3


def test_vector_store_satisfies_protocol(store):
    """isinstance(store, VectorStore) is True."""
    assert isinstance(store, VectorStore)


# ---------------------------------------------------------------------------
# count_by_video tests (IDX-05)
# ---------------------------------------------------------------------------


def test_count_by_video_returns_zero_empty(store):
    """count_by_video returns 0 for nonexistent video on empty table."""
    assert store.count_by_video("nonexistent") == 0


def test_count_by_video_returns_correct_count(store):
    """count_by_video returns per-video row count after upserting mixed rows."""
    rows = [
        _make_row(video_id="video_A", chunk_index=0),
        _make_row(video_id="video_A", chunk_index=1),
        _make_row(video_id="video_A", chunk_index=2),
        _make_row(video_id="video_B", chunk_index=0),
        _make_row(video_id="video_B", chunk_index=1),
    ]
    store.upsert(rows)
    assert store.count_by_video("video_A") == 3
    assert store.count_by_video("video_B") == 2
    assert store.count_by_video("video_C") == 0
