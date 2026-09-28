import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Router", version="0.1.0")

ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "change-me")


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
    if not ROUTER_API_KEY:
        return
    if authorization != f"Bearer {ROUTER_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def providers():
    items = []

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        items.append({
            "id": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": groq_key,
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        })

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        items.append({
            "id": "gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": gemini_key,
            "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        })

    compat_base = os.getenv("OPENAI_COMPAT_BASE_URL")
    compat_key = os.getenv("OPENAI_COMPAT_API_KEY")
    compat_model = os.getenv("OPENAI_COMPAT_MODEL")
    if compat_base and compat_key and compat_model:
        items.append({
            "id": "openai-compat",
            "base_url": compat_base.rstrip("/"),
            "api_key": compat_key,
            "model": compat_model,
        })

    ollama_base = os.getenv("OLLAMA_BASE_URL")
    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_base and ollama_model:
        items.append({
            "id": "ollama",
            "base_url": ollama_base.rstrip("/"),
            "api_key": "ollama",
            "model": ollama_model,
        })

    return items


@app.get("/health")
async def health():
    return {"status": "ok", "providers": [p["id"] for p in providers()]}


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    check_auth(authorization)
    data = [{"id": "free/auto", "object": "model", "owned_by": "router"}]
    for p in providers():
        data.append({
            "id": f'{p["id"]}/{p["model"]}',
            "object": "model",
            "owned_by": p["id"],
        })
    return {"object": "list", "data": data}


async def call_provider(provider: dict, req: ChatRequest):
    payload = {
        "model": provider["model"],
        "messages": [m.model_dump() for m in req.messages],
        "stream": False,
    }

    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    headers = {
        "Authorization": f'Bearer {provider["api_key"]}',
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            f'{provider["base_url"]}/chat/completions',
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f'{provider["id"]} returned {response.status_code}: {response.text[:300]}'
        )

    result = response.json()
    result["router_provider"] = provider["id"]
    return result


@app.post("/v1/chat/completions")
async def chat(
    req: ChatRequest,
    authorization: str | None = Header(default=None),
):
    check_auth(authorization)

    available = providers()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No provider configured. Add at least one provider API key.",
        )

    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not enabled in this starter version.",
        )

    candidates = available

    if req.model != "free/auto":
        matches = []
        for p in available:
            full_name = f'{p["id"]}/{p["model"]}'
            if req.model == full_name or req.model == p["id"] or req.model == p["model"]:
                matches.append(p)
        if not matches:
            raise HTTPException(status_code=404, detail="Requested model is not configured")
        candidates = matches

    errors = []

    for provider in candidates:
        try:
            return await call_provider(provider, req)
        except Exception as exc:
            errors.append({"provider": provider["id"], "error": str(exc)})

    raise HTTPException(
        status_code=502,
        detail={"message": "All configured providers failed", "errors": errors},
    )
