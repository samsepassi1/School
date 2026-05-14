"""Gradio chat front-end.

Multimodal chat UI where a trainee agent talks to a simulated angry customer.
Every outgoing agent message (text or file) is moderated first. If anything
is flagged, the message is *blocked* and the moderation rationale is shown
instead of being delivered to the customer. Otherwise the message is sent
and the customer agent replies.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any

import gradio as gr

from agents.audio_agent import moderate_audio
from agents.customer_agent import customer_agent
from agents.image_agent import moderate_image
from agents.text_agent import moderate_text
from agents.video_agent import moderate_video
from tracing import tracer
from schemas.moderation_result import ModerationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_media(path: str | os.PathLike[str]) -> str:
    """Return one of {"image", "audio", "video"} for a given file path."""
    guess, _ = mimetypes.guess_type(str(path))
    if guess is None:
        suffix = Path(path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
            return "image"
        if suffix in {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}:
            return "audio"
        if suffix in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
            return "video"
        return "image"
    if guess.startswith("image/"):
        return "image"
    if guess.startswith("audio/"):
        return "audio"
    if guess.startswith("video/"):
        return "video"
    return "image"


async def _moderate_file(path: str) -> tuple[str, ModerationResult]:
    """Pick the right moderation agent for ``path`` and run it on the bytes."""
    kind = _classify_media(path)
    data = Path(path).read_bytes()
    filename = Path(path).name
    if kind == "image":
        result: ModerationResult = await moderate_image(data, filename=filename)
    elif kind == "audio":
        result = await moderate_audio(data, filename=filename)
    else:
        result = await moderate_video(data, filename=filename)
    return kind, result


def _format_block(result: ModerationResult, what: str) -> str:
    """Human-readable block message for the chat UI."""
    flags = []
    if result.contains_pii:
        flags.append("contains PII")
    if result.is_unfriendly:
        flags.append("unfriendly tone")
    if result.is_unprofessional:
        flags.append("unprofessional tone")
    for extra in ("is_disturbing", "is_low_quality"):
        if getattr(result, extra, False):
            flags.append(extra.removeprefix("is_").replace("_", " "))
    flags_str = ", ".join(flags) if flags else "policy violation"
    return (
        f"⚠️ Your {what} was blocked by moderation ({flags_str}).\n\n"
        f"Rationale: {result.rationale}"
    )


# ---------------------------------------------------------------------------
# Chat callback
# ---------------------------------------------------------------------------


def _new_session_id() -> str:
    return f"conv-{uuid.uuid4().hex[:12]}"


async def respond(
    message: dict[str, Any],
    history: list[dict[str, Any]],
    session_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Main chat handler.

    Moderates each user turn (text and any uploaded files), and if everything
    passes, asks ``customer_agent`` for a reply. Returns the updated history
    and the session id.
    """
    if not session_id:
        session_id = _new_session_id()

    history = list(history or [])
    text = (message or {}).get("text") or ""
    files = list((message or {}).get("files") or [])

    with tracer.start_as_current_span("chat_turn") as turn_span:
        turn_span.set_attribute("session.id", session_id)
        turn_span.set_attribute("chat.has_text", bool(text.strip()))
        turn_span.set_attribute("chat.num_files", len(files))

        blocked = False
        block_reasons: list[str] = []

        # --- Moderate text ----------------------------------------------------
        if text.strip():
            with tracer.start_as_current_span("moderate_text") as span:
                span.set_attribute("session.id", session_id)
                span.set_attribute("moderation.media", "text")
                text_result = await moderate_text(text)
                span.set_attribute("moderation.flagged", text_result.flagged)
                span.set_attribute("moderation.rationale", text_result.rationale)

            history.append({"role": "user", "content": text})
            if text_result.flagged:
                blocked = True
                block_reasons.append(_format_block(text_result, "message"))

        # --- Moderate files ---------------------------------------------------
        for file_path in files:
            history.append({"role": "user", "content": {"path": file_path}})
            with tracer.start_as_current_span("moderate_file") as span:
                span.set_attribute("session.id", session_id)
                kind, file_result = await _moderate_file(file_path)
                span.set_attribute("moderation.media", kind)
                span.set_attribute("moderation.flagged", file_result.flagged)
                span.set_attribute("moderation.rationale", file_result.rationale)
            if file_result.flagged:
                blocked = True
                block_reasons.append(_format_block(file_result, kind))

        # --- Decide what to do ------------------------------------------------
        if blocked:
            for reason in block_reasons:
                history.append({"role": "assistant", "content": reason})
            turn_span.set_attribute("chat.blocked", True)
            return history, session_id

        turn_span.set_attribute("chat.blocked", False)

        # Build a transcript so the customer agent has conversation context
        transcript_lines: list[str] = []
        for msg in history:
            role = "Agent" if msg["role"] == "user" else "Customer"
            content = msg["content"]
            if isinstance(content, dict):
                content = f"[{Path(content.get('path', 'file')).name}]"
            transcript_lines.append(f"{role}: {content}")
        prompt = (
            "Continue the conversation as the angry customer. Reply with only "
            "your next line.\n\n" + "\n".join(transcript_lines)
        )

        with tracer.start_as_current_span("customer_reply") as span:
            span.set_attribute("session.id", session_id)
            reply = await customer_agent.run(prompt)
            customer_text = reply.output if hasattr(reply, "output") else str(reply)

        history.append({"role": "assistant", "content": customer_text})
        return history, session_id


def respond_sync(
    message: dict[str, Any],
    history: list[dict[str, Any]],
    session_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Sync wrapper so Gradio can call the async handler."""
    return asyncio.run(respond(message, history, session_id))


# ---------------------------------------------------------------------------
# Feedback + conversation lifecycle
# ---------------------------------------------------------------------------


def submit_feedback(
    feedback_text: str,
    session_id: str,
    history: list[dict[str, Any]],
) -> str:
    """Record free-text feedback as a tracing span."""
    with tracer.start_as_current_span("feedback") as span:
        span.set_attribute("session.id", session_id or "unknown")
        span.set_attribute("feedback.content", feedback_text or "")
        span.set_attribute("feedback.turns", len(history or []))
    return "Thanks — feedback recorded."


def end_conversation(
    session_id: str, history: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, str]:
    """End the conversation: emit a ``conversation`` span and reset state."""
    with tracer.start_as_current_span("conversation") as span:
        span.set_attribute("session.id", session_id or "unknown")
        span.set_attribute("conversation.turns", len(history or []))
        span.set_attribute("conversation.ended", True)
    return [], _new_session_id(), "Conversation ended. A new session is ready."


# ---------------------------------------------------------------------------
# Gradio Blocks layout
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ACME Customer Service Trainer") as demo:
        gr.Markdown(
            "# ACME Customer Service Trainer\n"
            "Chat with a simulated angry customer. Every message you send is "
            "moderated for PII, unfriendly tone, unprofessional tone and "
            "unsafe media before it reaches the customer."
        )

        session_state = gr.State(value=_new_session_id())

        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
        )

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_types=["image", "audio", "video"],
            placeholder="Type a message, or attach an image / audio / video…",
            show_label=False,
        )

        with gr.Row():
            end_btn = gr.Button("End Conversation", variant="stop")
            status_box = gr.Markdown("")

        with gr.Accordion("Leave feedback", open=False):
            feedback_input = gr.Textbox(
                label="Feedback",
                placeholder="How did this turn go? What would you improve?",
                lines=2,
            )
            feedback_btn = gr.Button("Send feedback")
            feedback_status = gr.Markdown("")

        chat_input.submit(
            fn=respond_sync,
            inputs=[chat_input, chatbot, session_state],
            outputs=[chatbot, session_state],
        ).then(
            fn=lambda: {"text": "", "files": []},
            inputs=None,
            outputs=chat_input,
        )

        end_btn.click(
            fn=end_conversation,
            inputs=[session_state, chatbot],
            outputs=[chatbot, session_state, status_box],
        )

        feedback_btn.click(
            fn=submit_feedback,
            inputs=[feedback_input, session_state, chatbot],
            outputs=feedback_status,
        )

    return demo


demo = build_demo()


def main() -> None:
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
