import os
import time
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Router", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "").strip()


class Message(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str = "free/auto"
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


def check_auth(authorization: str | None):
    if ROUTER_API_KEY and authorization != f"Bearer {ROUTER_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def providers():
    items = []
    if key := os.getenv("GROQ_API_KEY"):
        items.append({"id": "groq", "base_url": "https://api.groq.com/openai/v1",
                      "api_key": key, "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")})
    if key := os.getenv("GEMINI_API_KEY"):
        items.append({"id": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                      "api_key": key, "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash")})
    base, key, model = (os.getenv("OPENAI_COMPAT_BASE_URL"), os.getenv("OPENAI_COMPAT_API_KEY"),
                        os.getenv("OPENAI_COMPAT_MODEL"))
    if base and key and model:
        items.append({"id": "openai-compat", "base_url": base.rstrip("/"),
                      "api_key": key, "model": model})
    base, model = os.getenv("OLLAMA_BASE_URL"), os.getenv("OLLAMA_MODEL")
    if base and model:
        items.append({"id": "ollama", "base_url": base.rstrip("/"),
                      "api_key": os.getenv("OLLAMA_API_KEY", "ollama"), "model": model})
    return items


@app.get("/")
async def root():
    return {"name": "Router", "version": "0.2.0", "status": "online",
            "docs": "/docs", "health": "/health", "models": "/v1/models"}


@app.get("/health")
async def health():
    ps = providers()
    return {"status": "ok", "configured_providers": [p["id"] for p in ps],
            "provider_count": len(ps)}


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    check_auth(authorization)
    data = [{"id": "free/auto", "object": "model", "owned_by": "router"}]
    data += [{"id": f'{p["id"]}/{p["model"]}', "object": "model", "owned_by": p["id"]}
             for p in providers()]
    return {"object": "list", "data": data}


async def call_provider(provider: dict, req: ChatRequest):
    payload = {"model": provider["model"],
               "messages": [m.model_dump() for m in req.messages],
               "stream": False}
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    headers = {"Authorization": f'Bearer {provider["api_key"]}',
               "Content-Type": "application/json"}
    timeout = httpx.Timeout(90.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f'{provider["base_url"]}/chat/completions',
                              headers=headers, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f'{provider["id"]} HTTP {r.status_code}: {r.text[:500]}')
    result = r.json()
    result.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
    result.setdefault("object", "chat.completion")
    result.setdefault("created", int(time.time()))
    result["router_provider"] = provider["id"]
    result["router_model"] = provider["model"]
    return result


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not available yet; use stream=false.")

    available = providers()
    if not available:
        raise HTTPException(status_code=503, detail={
            "message": "Router is online but no inference provider is configured.",
            "fix": "Set GROQ_API_KEY, GEMINI_API_KEY, an OpenAI-compatible provider, or Ollama."
        })

    candidates = available
    if req.model != "free/auto":
        candidates = [p for p in available if req.model in
                      (p["id"], p["model"], f'{p["id"]}/{p["model"]}')]
        if not candidates:
            raise HTTPException(status_code=404, detail="Requested model is not configured.")

    errors = []
    for provider in candidates:
        try:
            return await call_provider(provider, req)
        except Exception as exc:
            errors.append({"provider": provider["id"], "error": str(exc)})

    raise HTTPException(status_code=502, detail={
        "message": "All configured providers failed.", "errors": errors
    })
