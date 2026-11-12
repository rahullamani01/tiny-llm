"""Tests for the streaming chat server core and FastAPI app."""
import json
import string

import torch

from chat.template import CHAT_SPECIAL_TOKENS
from model.model import TinyGPT
from server.chat import iter_chat_events, validate_sampling
from tokenizer import CharTokenizer

try:
    from fastapi.testclient import TestClient

    from server.app import create_app
except (ImportError, SystemExit):  # server extras are optional for the test suite
    TestClient = None
    create_app = None


def _model_and_tok():
    text = string.ascii_letters + string.digits + " .,!?'\n"
    tok = CharTokenizer(text, special_tokens=CHAT_SPECIAL_TOKENS)
    torch.manual_seed(0)
    model = TinyGPT(
        vocab_size=tok.vocab_size,
        block_size=16,
        n_embd=32,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    ).eval()
    return model, tok


def test_sampling_validation():
    validate_sampling()
    for kwargs in (
        dict(max_new_tokens=0),
        dict(temperature=-0.1),
        dict(top_k=0),
        dict(top_p=0.0),
        dict(top_p=1.1),
        dict(repetition_penalty=0.0),
    ):
        try:
            validate_sampling(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_stream_events_end_with_done_content():
    model, tok = _model_and_tok()
    events = list(
        iter_chat_events(
            model,
            tok,
            [{"role": "user", "content": "hi"}],
            system="Be nice.",
            max_new_tokens=4,
            temperature=0.0,
            use_cache=False,
        )
    )
    assert events[-1]["done"] is True
    tokens = [event["token"] for event in events if "token" in event]
    assert events[-1]["content"] == "".join(tokens)
    assert "<|" not in events[-1]["content"]


def test_fastapi_health_chat_and_errors():
    if create_app is None:
        return  # fastapi/httpx not installed; core tests above still run

    model, tok = _model_and_tok()
    client = TestClient(create_app(model, tok, system="Be nice.", use_cache=False))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["parameters"] > 0

    page = client.get("/")
    assert page.status_code == 200
    assert "tiny-llm chat" in page.text
    assert client.get("/static/app.js").status_code == 200

    response = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_new_tokens": 3,
            "temperature": 0.0,
            "use_cache": False,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1]["done"] is True

    bad = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": -1,
        },
    )
    assert bad.status_code == 400

    empty = client.post("/api/chat", json={"messages": []})
    assert empty.status_code == 400


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
