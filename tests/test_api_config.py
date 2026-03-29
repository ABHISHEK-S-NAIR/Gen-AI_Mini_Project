from fastapi.testclient import TestClient

from app.main import app
from app.core.state import state


client = TestClient(app)


def setup_function() -> None:
    state.clear()


def test_config_patch_before_ingestion_allowed() -> None:
    response = client.patch("/config", json={"chunk_size": 300, "chunk_overlap": 30})
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_size"] == 300
    assert body["chunk_overlap"] == 30


def test_task_without_ingestion_fails() -> None:
    response = client.post("/task", json={"task": "analyse", "query": "transformer"})
    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "NO_PAPERS_INGESTED"


def test_ask_requires_question() -> None:
    state.papers["p1"] = type("Obj", (), {"paper_id": "p1", "filename": "a.pdf", "raw_text": "x"})()
    state.paper_embeddings["p1"] = [0.0] * 128
    response = client.post("/task", json={"task": "ask"})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "E011"
