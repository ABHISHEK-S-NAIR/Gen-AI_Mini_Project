from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ui_root_serves_html() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "PaperMind Control Deck" in response.text


def test_web_static_assets_served() -> None:
    response = client.get("/web/style.css")
    assert response.status_code == 200
    assert "--bg" in response.text
