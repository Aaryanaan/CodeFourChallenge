"""Tests for LanceVectorStore (IDX-02, IDX-04)."""
import pytest

from videosearch.vector_store import LanceVectorStore

VECTOR_DIM = 768


@pytest.fixture
def store(tmp_path):
    return LanceVectorStore(index_dir=tmp_path)


@pytest.fixture
def sample_row():
    return {
        "vector": [0.1] * VECTOR_DIM,
        "video_id": "vid_001",
        "chunk_index": 0,
        "start_time": 0.0,
        "end_time": 30.0,
        "duration": 30.0,
        "combined_text": "Transcript: hello world\nOCR: STOP",
        "volume_level": "normal",
        "has_speech": True,
        "has_ocr": True,
        "has_raised_voice": False,
        "scene_type": "detected",
    }


def test_table_schema(store):
    table = store._get_table()
    schema = table.schema
    field_names = [f.name for f in schema]
    assert "vector" in field_names
    assert "video_id" in field_names
    assert "chunk_index" in field_names
    assert "volume_level" in field_names
    assert "has_speech" in field_names
    assert "has_ocr" in field_names
    assert "has_raised_voice" in field_names
    assert "duration" in field_names


def test_upsert_roundtrip(store, sample_row):
    store.upsert([sample_row])
    store.upsert([sample_row])  # Re-upsert same row
    results = store.search(sample_row["vector"], top_k=10)
    # Should have exactly 1 row (upsert replaced, not duplicated)
    matching = [r for r in results if r["video_id"] == "vid_001" and r["chunk_index"] == 0]
    assert len(matching) == 1


def test_search_returns_results(store, sample_row):
    store.upsert([sample_row])
    results = store.search(sample_row["vector"], top_k=5)
    assert len(results) >= 1
    assert "_distance" in results[0]
    assert results[0]["video_id"] == "vid_001"


def test_filter_by_raised_voice(store):
    row_calm = {
        "vector": [0.1] * VECTOR_DIM,
        "video_id": "vid_001", "chunk_index": 0,
        "start_time": 0.0, "end_time": 30.0, "duration": 30.0,
        "combined_text": "calm speech",
        "volume_level": "normal", "has_speech": True,
        "has_ocr": False, "has_raised_voice": False, "scene_type": "detected",
    }
    row_loud = {
        "vector": [0.2] * VECTOR_DIM,
        "video_id": "vid_001", "chunk_index": 1,
        "start_time": 30.0, "end_time": 60.0, "duration": 30.0,
        "combined_text": "loud speech",
        "volume_level": "loud", "has_speech": True,
        "has_ocr": False, "has_raised_voice": True, "scene_type": "detected",
    }
    store.upsert([row_calm, row_loud])
    results = store.search(
        [0.2] * VECTOR_DIM, top_k=10,
        filter_expr="has_raised_voice = true",
    )
    assert all(r["has_raised_voice"] for r in results)


def test_filter_by_duration(store):
    short = {
        "vector": [0.1] * VECTOR_DIM,
        "video_id": "vid_001", "chunk_index": 0,
        "start_time": 0.0, "end_time": 10.0, "duration": 10.0,
        "combined_text": "short",
        "volume_level": "normal", "has_speech": True,
        "has_ocr": False, "has_raised_voice": False, "scene_type": "detected",
    }
    long_chunk = {
        "vector": [0.2] * VECTOR_DIM,
        "video_id": "vid_001", "chunk_index": 1,
        "start_time": 10.0, "end_time": 55.0, "duration": 45.0,
        "combined_text": "long chunk",
        "volume_level": "normal", "has_speech": True,
        "has_ocr": False, "has_raised_voice": False, "scene_type": "detected",
    }
    store.upsert([short, long_chunk])
    results = store.search(
        [0.2] * VECTOR_DIM, top_k=10,
        filter_expr="duration >= 30.0",
    )
    assert all(r["duration"] >= 30.0 for r in results)


def test_search_with_no_filter(store, sample_row):
    store.upsert([sample_row])
    results = store.search(sample_row["vector"], top_k=5)
    assert len(results) >= 1


def test_vector_store_satisfies_protocol(tmp_path):
    from videosearch.protocols import VectorStore
    vs = LanceVectorStore(index_dir=tmp_path)
    assert isinstance(vs, VectorStore)
