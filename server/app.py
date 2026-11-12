"""FastAPI chat server + static web UI.

Run:
    python -m server.app --ckpt checkpoints/tinygpt-chat.pt
    # open http://127.0.0.1:8000

The API is intentionally small:
    POST /api/chat   -> Server-Sent Events stream
    GET  /api/health -> model metadata
"""
import argparse
import json
import logging
import threading
from pathlib import Path
from typing import Literal

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover - exercised only without deps
    raise SystemExit(
        "error: the web server needs FastAPI and Uvicorn.\n"
        "install them with: pip install fastapi uvicorn"
    ) from e

from chat.template import CHAT_SPECIAL_TOKENS, build_chat_prompt
from inference.chat import DEFAULT_SYSTEM
from inference.generate import load_model
from server.chat import iter_chat_events, validate_sampling
from utils import get_device

logger = logging.getLogger("tiny_llm.server")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message]
    system: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float | None = None
    repetition_penalty: float = 1.1
    use_cache: bool = True
    seed: int | None = None


def _sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(model, tok, system=DEFAULT_SYSTEM, use_cache=True):
    app = FastAPI(title="tiny-llm chat", version="0.1.0")
    generation_lock = threading.Lock()

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        path = WEB_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=500, detail="web/index.html is missing")
        return FileResponse(path)

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "parameters": sum(p.numel() for p in model.parameters()),
            "context": model.block_size,
            "pos_embedding": model.pos_embedding,
            "vocab_size": tok.vocab_size,
            "special_tokens": list(tok.special_tokens),
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        system_prompt = system if req.system is None else req.system
        try:
            validate_sampling(
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            )
            build_chat_prompt(
                messages,
                tok,
                model.block_size,
                add_generation_prompt=True,
                system=system_prompt,
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

        def event_stream():
            # Serialize generation: one tiny model, one shared set of weights.
            with generation_lock:
                try:
                    for event in iter_chat_events(
                        model,
                        tok,
                        messages,
                        system=system_prompt,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        repetition_penalty=req.repetition_penalty,
                        use_cache=req.use_cache and use_cache,
                        seed=req.seed,
                    ):
                        yield _sse(event)
                except Exception as e:  # keep the stream well-formed
                    logger.exception("chat generation failed")
                    yield _sse({"error": str(e)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Serve the tiny-llm chat UI.")
    p.add_argument("--ckpt", default="checkpoints/tinygpt-chat.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--no-kv-cache", action="store_true",
                   help="disable KV caching by default")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        model, tok = load_model(args.ckpt, get_device(args.device))
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"error: could not load {args.ckpt}: {e}")

    missing = [t for t in CHAT_SPECIAL_TOKENS if t not in tok.special_tokens]
    if missing:
        raise SystemExit(
            f"error: {args.ckpt} is a completion checkpoint, not an SFT chat "
            f"checkpoint (missing {', '.join(missing)})"
        )

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "error: uvicorn is not installed; run: pip install fastapi uvicorn"
        ) from e

    app = create_app(model, tok, system=args.system, use_cache=not args.no_kv_cache)
    print(f"serving {args.ckpt} on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
