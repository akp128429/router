import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AKP Router API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "").strip()
USAGE = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "errors": 0})


class Message(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str = "free/auto"
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


def check_auth(authorization: str | None):
    if ROUTER_API_KEY and authorization != f"Bearer {ROUTER_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def providers():
    items = []
    if key := os.getenv("GROQ_API_KEY"):
        items.append({"id": "groq", "base_url": "https://api.groq.com/openai/v1", "api_key": key,
                      "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                      "capabilities": ["chat", "stream", "tools"]})
    if key := os.getenv("GEMINI_API_KEY"):
        items.append({"id": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key": key,
                      "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                      "capabilities": ["chat", "stream", "tools", "vision"]})
    base, key, model = os.getenv("OPENAI_COMPAT_BASE_URL"), os.getenv("OPENAI_COMPAT_API_KEY"), os.getenv("OPENAI_COMPAT_MODEL")
    if base and key and model:
        items.append({"id": "openai-compat", "base_url": base.rstrip("/"), "api_key": key, "model": model,
                      "capabilities": ["chat", "stream", "tools"]})
    base, model = os.getenv("OLLAMA_BASE_URL"), os.getenv("OLLAMA_MODEL")
    if base and model:
        items.append({"id": "ollama", "base_url": base.rstrip("/"), "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
                      "model": model, "capabilities": ["chat", "stream"]})
    return items


def candidates_for(model: str):
    ps = providers()
    if model == "free/auto":
        return ps
    return [p for p in ps if model in (p["id"], p["model"], f'{p["id"]}/{p["model"]}')]


def payload_for(provider: dict, req: ChatRequest, stream: bool):
    payload: dict[str, Any] = {"model": provider["model"], "messages": [m.model_dump() for m in req.messages], "stream": stream}
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.tools is not None:
        payload["tools"] = req.tools
    if req.tool_choice is not None:
        payload["tool_choice"] = req.tool_choice
    return payload


def headers_for(provider: dict):
    return {"Authorization": f'Bearer {provider["api_key"]}', "Content-Type": "application/json"}


def add_usage(provider_id: str, result: dict):
    USAGE[provider_id]["requests"] += 1
    usage = result.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        USAGE[provider_id][key] += int(usage.get(key) or 0)


async def call_provider(provider: dict, req: ChatRequest):
    timeout = httpx.Timeout(90.0, connect=15.0)
    last_error = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f'{provider["base_url"]}/chat/completions', headers=headers_for(provider),
                                      json=payload_for(provider, req, False))
            if r.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                await asyncio.sleep(0.5)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f'{provider["id"]} HTTP {r.status_code}: {r.text[:500]}')
            result = r.json()
            result.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
            result.setdefault("object", "chat.completion")
            result.setdefault("created", int(time.time()))
            result["router_provider"] = provider["id"]
            result["router_model"] = provider["model"]
            add_usage(provider["id"], result)
            return result
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                await asyncio.sleep(0.5)
    USAGE[provider["id"]]["errors"] += 1
    raise last_error or RuntimeError("Provider failed")


async def stream_provider(provider: dict, req: ChatRequest, request_id: str) -> AsyncIterator[str]:
    timeout = httpx.Timeout(90.0, connect=15.0, read=90.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f'{provider["base_url"]}/chat/completions',
                                     headers=headers_for(provider), json=payload_for(provider, req, True)) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="replace")[:500]
                    raise RuntimeError(f'{provider["id"]} HTTP {r.status_code}: {body}')
                USAGE[provider["id"]]["requests"] += 1
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        try:
                            obj = json.loads(data)
                            obj["router_provider"] = provider["id"]
                            obj["router_request_id"] = request_id
                            yield f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"
                        except json.JSONDecodeError:
                            yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
    except Exception as exc:
        USAGE[provider["id"]]["errors"] += 1
        err = {"error": {"message": str(exc), "type": "router_upstream_error"}, "router_request_id": request_id}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/")
async def root():
    return {"name": "AKP Router API", "version": "0.3.0", "status": "online", "docs": "/docs",
            "health": "/health", "models": "/v1/models", "providers": "/v1/providers", "usage": "/v1/usage"}


@app.get("/health")
async def health():
    ps = providers()
    return {"status": "ok", "configured_providers": [p["id"] for p in ps], "provider_count": len(ps), "version": "0.3.0"}


@app.get("/v1/providers")
async def provider_status(authorization: str | None = Header(default=None)):
    check_auth(authorization)
    return {"object": "list", "data": [{"id": p["id"], "model": p["model"], "capabilities": p["capabilities"],
                                        "configured": True} for p in providers()]}


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    check_auth(authorization)
    data = [{"id": "free/auto", "object": "model", "owned_by": "router", "capabilities": ["chat", "stream"]}]
    data += [{"id": f'{p["id"]}/{p["model"]}', "object": "model", "owned_by": p["id"],
              "capabilities": p["capabilities"]} for p in providers()]
    return {"object": "list", "data": data}


@app.get("/v1/usage")
async def usage(authorization: str | None = Header(default=None)):
    check_auth(authorization)
    return {"object": "usage", "providers": dict(USAGE), "note": "In-memory counters reset when the service restarts."}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    available = providers()
    if not available:
        raise HTTPException(status_code=503, detail={"message": "Router is online but no inference provider is configured.",
            "fix": "Set GROQ_API_KEY, GEMINI_API_KEY, an OpenAI-compatible provider, or Ollama."})
    candidates = candidates_for(req.model)
    if not candidates:
        raise HTTPException(status_code=404, detail="Requested model is not configured.")

    if req.stream:
        # Streaming cannot safely fail over after bytes have been sent, so use the first selected provider.
        return StreamingResponse(stream_provider(candidates[0], req, request.state.request_id),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    errors = []
    for provider in candidates:
        try:
            result = await call_provider(provider, req)
            result["router_request_id"] = request.state.request_id
            return result
        except Exception as exc:
            errors.append({"provider": provider["id"], "error": str(exc)})
    raise HTTPException(status_code=502, detail={"message": "All configured providers failed.", "errors": errors})
