# Router

A small OpenAI-compatible, free-first multi-model API gateway.

## What it supports

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- Free-first routing across configured providers
- Gemini, Groq and OpenAI-compatible endpoints
- Optional local/self-hosted endpoints such as Ollama
- Provider fallback when a request fails
- Simple bearer-token protection for your public gateway

> This project does not bypass provider billing, quotas, authentication, or terms. Proprietary providers still require legitimate API access.

## Quick start

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl http://localhost:8000/health
```

### Chat completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me" \
  -d '{
    "model": "free/auto",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

## Environment variables

See `.env.example`.

## Routing

`free/auto` tries enabled providers in this order:

1. Groq
2. Gemini
3. GitHub/OpenAI-compatible endpoint
4. Ollama/local endpoint

You can also directly request one of the configured model aliases.

## Deploy

The repository includes a `render.yaml` for simple Render deployment. Add your API keys as environment variables in the hosting dashboard.

## Future adapters

Image, video and music generation can be exposed as separate endpoints, but each provider must be integrated using an official or otherwise authorized API. Open-source/self-hosted music or video models can also be added.
