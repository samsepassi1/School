# ACME Multimodal Content Moderation

AI-powered content moderation for customer-service interactions at the
(fictional) ACME Enterprise. A trainee customer-service agent chats with a
simulated angry customer (played by an LLM), and every message — text,
image, audio, video — is screened in real time for:

- PII (personally identifiable information)
- unfriendly / rude tone
- unprofessional language
- disturbing / unsafe visuals
- low-quality media

Blocked messages are rejected with a structured rationale instead of being
delivered to the customer.

## Architecture

| Component          | File                          | Notes                                                            |
|--------------------|-------------------------------|------------------------------------------------------------------|
| Text moderator     | `agents/text_agent.py`        | Pydantic-AI + Gemini, returns `ModerationResult`                 |
| Image moderator    | `agents/image_agent.py`       | Pydantic-AI multimodal (`BinaryContent`), `ImageModerationResult`|
| Video moderator    | `agents/video_agent.py`       | Same shape as image                                              |
| Audio moderator    | `agents/audio_agent.py`       | `AudioModerationResult` with `is_low_quality`                    |
| Customer simulator | `agents/customer_agent.py`    | Plays an angry customer                                          |
| Result schemas     | `types/moderation_result.py`  | Pydantic models shared by all agents                             |
| Tracing            | `tracing.py`                  | OpenTelemetry → Phoenix, exposes a module-level `tracer`         |
| Backend            | `fastapi_app.py`              | REST API at `/api/v1/moderate-{text,image,video,audio}`          |
| Frontend           | `gradio_app.py`               | Multimodal chat UI                                               |
| Entrypoint         | `main.py`                     | `uv run multimodal-moderation` runs Phoenix + FastAPI + Gradio   |

## Run it

```bash
cp .env.example .env  # then fill in your keys
uv sync
uv run multimodal-moderation
```

Then open:

- http://localhost:7860/  — Gradio chat UI
- http://localhost:8000/docs — FastAPI Swagger
- http://localhost:6006/  — Phoenix traces

## Test it

```bash
uv run tests/test_moderation_result.py -vv
uv run tests/test_text_agent.py -vv
uv run tests/test_image_agent.py -vv
uv run tests/test_video_agent.py -vv
uv run tests/test_audio_agent.py -vv
uv run tests/test_gradio_app.py -vv
uv run pytest tests/ -vv
```

Tests use `pydantic_ai.models.function.FunctionModel` to override the LLM,
so they pass without making a network call.

## Evaluate it

```bash
uv run evals/text/test_cases.py
uv run evals/image/test_cases.py
uv run evals/audio/test_cases.py
uv run evals/video/test_cases.py
```

Drop your own test assets in `evals/test_data/`. See the README in that
directory for the expected filenames.
