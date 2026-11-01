"""
Thin wrappers around model APIs that coerce a model's response into a
Pydantic schema. Keeping these generic (not agent-specific) so any agent can
reuse them with its own model and schema.

Three entry points, not one function with branches inside: Ollama, Groq
text, and Groq vision each need genuinely different handling (see each
function's docstring), and folding them into one function would hide that
they're solving different problems rather than sharing one mechanism.
structured_chat_groq_vision lives next to structured_chat_groq rather than
duplicating its error handling from scratch, since the two share Groq's
response_format=json_schema mechanism and only differ in message shape.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Type, TypeVar

import ollama
from groq import AsyncGroq, RateLimitError
from pydantic import BaseModel

from observability import observability as obs
from app.core.config import settings

T = TypeVar("T", bound=BaseModel)


class LLMExtractionError(Exception):
    """Raised when the model output can't be parsed/validated against the schema."""


class LLMRateLimitError(LLMExtractionError):
    """
    Raised specifically for a 429/rate-limit response, as distinct from
    other LLMExtractionErrors (parse failures, network errors, malformed
    output). Callers that retry blindly on LLMExtractionError will retry a
    rate limit exactly as fast as everything else - but an immediate retry
    inside the same per-minute window is guaranteed to hit the same limit
    again, so callers need to be able to tell this case apart and back off
    instead of retrying instantly. See structured_chat_groq_vision's
    docstring.
    """


# ---------------------------------------------------------------------------
# Tracing helpers (Langfuse). Provider-specific parsing lives here, next to
# the calls it describes; observability/observability.py stays provider-agnostic.
# Everything below degrades to {} / None on an unexpected response shape -
# telemetry must never be the reason a model call fails.
# ---------------------------------------------------------------------------


def _field(obj, key):
    """ollama's ChatResponse is attribute- AND subscript-accessible; groq's
    objects are attribute-only. Try both, never raise."""
    value = getattr(obj, key, None)
    if value is None:
        try:
            value = obj[key]
        except Exception:  # noqa: BLE001
            value = None
    return value


def _ollama_telemetry(response) -> tuple[dict, dict, str | None]:
    """(usage, metadata, done_reason). Durations arrive in nanoseconds.
    `load_s` well above zero means the model was (re)loaded for this call -
    a swap-in, which is what a shared 6GB GPU makes expensive."""
    prompt_tokens = _field(response, "prompt_eval_count")
    output_tokens = _field(response, "eval_count")
    usage = {}
    if isinstance(prompt_tokens, int):
        usage["input"] = prompt_tokens
    if isinstance(output_tokens, int):
        usage["output"] = output_tokens
    if usage:
        usage["total"] = sum(usage.values())

    def seconds(key):
        value = _field(response, key)
        return round(value / 1e9, 2) if isinstance(value, (int, float)) else None

    eval_s = seconds("eval_duration")
    done_reason = _field(response, "done_reason")
    meta = {
        "done_reason": done_reason,
        "total_s": seconds("total_duration"),
        "load_s": seconds("load_duration"),
        "prompt_eval_s": seconds("prompt_eval_duration"),
        "eval_s": eval_s,
    }
    if isinstance(output_tokens, int) and eval_s:
        meta["tokens_per_second"] = round(output_tokens / eval_s, 1)
    return usage, {k: v for k, v in meta.items() if v is not None}, done_reason


def _groq_telemetry(response) -> tuple[dict, dict, str | None]:
    """(usage, metadata, finish_reason). `finish_reason == "length"` means
    the output hit max_tokens - the usual cause of truncated, unparseable
    JSON, so it's surfaced as a warning on the span."""
    usage = {}
    meta: dict = {}
    raw_usage = getattr(response, "usage", None)
    if raw_usage is not None:
        for ours, theirs in (("input", "prompt_tokens"), ("output", "completion_tokens"), ("total", "total_tokens")):
            value = getattr(raw_usage, theirs, None)
            if isinstance(value, int):
                usage[ours] = value
        for ours, theirs in (
            ("queue_time_s", "queue_time"),
            ("prompt_time_s", "prompt_time"),
            ("completion_time_s", "completion_time"),
            ("total_time_s", "total_time"),
        ):
            value = getattr(raw_usage, theirs, None)
            if isinstance(value, (int, float)):
                meta[ours] = round(float(value), 3)
    choices = getattr(response, "choices", None) or []
    finish = getattr(choices[0], "finish_reason", None) if choices else None
    meta["finish_reason"] = finish
    meta["response_model"] = getattr(response, "model", None)
    return usage, {k: v for k, v in meta.items() if v is not None}, finish



def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _example_from_json_schema(node: dict, defs: dict) -> object:
    """
    Recursively build one plausible placeholder value for a JSON-schema
    node, resolving $ref/$defs (nested Pydantic sub-models) and
    anyOf/Optional wrapping along the way. Used to build a flat, filled
    *instance* of a schema rather than handing a smaller model the raw
    schema definition - see _build_example_json's docstring for why that
    distinction actually matters here, not just cosmetically.
    """
    if "$ref" in node:
        ref_name = node["$ref"].split("/")[-1]
        return _example_from_json_schema(defs[ref_name], defs)

    if "anyOf" in node:
        for option in node["anyOf"]:
            if option.get("type") != "null":
                return _example_from_json_schema(option, defs)
        return None

    node_type = node.get("type")

    if node_type == "object":
        if "properties" in node:
            return {
                name: _example_from_json_schema(prop, defs)
                for name, prop in node["properties"].items()
            }
        if isinstance(node.get("additionalProperties"), dict):
            # dict[str, X] fields (e.g. `specifications: dict[str, str]`)
            return {"example_key": _example_from_json_schema(node["additionalProperties"], defs)}
        return {}

    if node_type == "array":
        return [_example_from_json_schema(node.get("items", {}), defs)]

    if node_type == "string":
        hint = node.get("description") or node.get("title") or "value"
        return f"<{hint[:60]}>"

    if node_type == "integer":
        return 1

    if node_type == "number":
        return 1.0

    if node_type == "boolean":
        return True

    return None


def _build_example_json(schema: Type[BaseModel]) -> str:
    """
    Build a filled-shape example instance from a Pydantic schema, instead of
    handing the model the raw JSON Schema. qwen3.5 (and likely other small
    models) sometimes echo the schema *definition* back verbatim -
    `properties`/`type`/`description` keys with real values nested inside
    `properties` - rather than returning a flat filled instance, because to a
    smaller model the schema definition and a filled instance look
    structurally similar. Showing a concrete example with placeholder values
    (not the schema metadata) removes that ambiguity entirely: there's no
    `properties` key for the model to imitate.

    Generic on purpose - this is used for every schema across every agent,
    not hand-written per schema, so a new agent gets this fix for free.
    """
    schema_dict = schema.model_json_schema()
    defs = schema_dict.get("$defs", {})
    example = _example_from_json_schema(schema_dict, defs)
    return json.dumps(example, indent=2)


def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction of a single JSON object from model output that may
    still contain stray commentary or leftover thinking-trace fragments
    around it, even with think disabled. Finds the first '{' and its
    brace-matched closing '}' rather than trusting the whole string is
    clean JSON.
    """
    text = _strip_code_fences(text)
    start = text.find("{")
    if start == -1:
        return text  # nothing to salvage - let json.loads raise a clear error

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced - let json.loads raise a clear error


def _encode_image_data_uri(path: str) -> str:
    """
    Reads a local file and encodes it as a base64 data URI. These are files
    this project generated (Agent 4's GeneratedImage.local_path), not remote
    URLs, so there's no image_url to pass through - the bytes have to be
    embedded directly.
    """
    ext = Path(path).suffix.lstrip(".").lower() or "png"
    media_type = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{media_type};base64,{b64}"


async def structured_chat(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    temperature: float = 0.2,
) -> T:
    """
    Call an Ollama model and coerce its response into `schema`.

    Why this doesn't just use Ollama's native `format=schema` structured
    output: qwen3.5 (a thinking-capable model) has two stacking bugs in
    Ollama as of mid-2026 (ollama/ollama#10929, #14645):
      - think left unset (defaults to True) + format=schema -> the model can
        burn its entire output budget on the hidden thinking trace and never
        emit the final JSON, returning EMPTY content. This is exactly the
        `Invalid JSON: EOF while parsing a value` / empty raw output failure.
      - think=False + format=schema -> the format constraint is silently
        IGNORED and the model returns free-form text instead of JSON.
    Net effect: native `format` isn't reliable on this model family right
    now. Workaround: disable thinking, drop `format`, describe the schema in
    the prompt instead, and parse the response ourselves with a tolerant
    extractor. Revisit once the upstream bugs are fixed - `format` is the
    more robust mechanism when it actually works.

    Also sets num_ctx explicitly: Ollama's runtime default context window
    (2048-4096 depending on version) is unrelated to a model's advertised
    max context, and research prompts here (up to ~20k chars of page text +
    8k of review text) will silently get truncated without this.

    Shows a filled-shape EXAMPLE instance in the prompt, not the raw JSON
    Schema - qwen3.5 (and likely other small models) will otherwise
    sometimes echo the schema definition back verbatim (`properties`/`type`/
    `description` keys, real values nested inside `properties`) instead of
    returning a flat filled instance, since the two look structurally
    similar to a smaller model. See _build_example_json's docstring.
    """
    client = ollama.AsyncClient(host=settings.ollama_host)

    schema_hint = (
        "\n\nRespond with ONLY a single JSON object - no markdown code fences, "
        "no commentary before or after it. Here is the exact shape to return, "
        "with placeholder values showing what kind of content belongs in each "
        "field - replace every placeholder with real content, and do not "
        "return this structure with the placeholders still in it, and do not "
        "return a JSON Schema definition (no 'properties'/'type' keys):\n"
        f"{_build_example_json(schema)}"
    )

    messages = [
        {"role": "system", "content": system_prompt + schema_hint},
        {"role": "user", "content": user_prompt},
    ]

    with obs.generation(
        name=f"ollama:{schema.__name__}",
        model=model,
        provider="ollama",
        input=messages,
        model_parameters={
            "temperature": temperature,
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": settings.ollama_num_predict,
            "keep_alive": "30m",
            "think": False,
        },
    ) as gen:
        response = await client.chat(
            model=model,
            messages=messages,
            think=False,
            keep_alive="30m",
            options={
                "temperature": temperature,
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": settings.ollama_num_predict,
            },
        )

        usage, meta, done_reason = _ollama_telemetry(response)
        if gen.active:
            meta.update(await obs.ollama_memory(settings.ollama_host, model))

        raw_content = response["message"]["content"] or ""

        if not raw_content.strip():
            # Some Ollama versions still leak the answer into the thinking field
            # even with think=False, depending on the model's chat template.
            raw_content = response["message"].get("thinking") or ""

        gen.record(
            output=raw_content,
            usage=usage,
            metadata=meta,
            warning="done_reason=length: output hit num_predict" if done_reason == "length" else None,
        )

        json_text = _extract_json_object(raw_content)

        try:
            parsed = json.loads(json_text)
            return schema.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001 - we want to wrap any parse/validation error
            raise LLMExtractionError(
                f"Model output failed schema validation: {exc}\nRaw output: {raw_content[:500]!r}"
            ) from exc


async def structured_chat_vision(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[str],
    schema: Type[T],
    temperature: float = 0.2,
    num_predict: int | None = None,
) -> T:
    """
    Vision-capable counterpart to structured_chat, for locally-run
    multimodal Ollama models - confirmed working for qwen3.5:4b (a real,
    accurate image description came back in testing), unlike Ollama's
    still-incomplete support for some other vision-packaged models whose
    separate mmproj files it doesn't yet load.

    Same reasoning as structured_chat for not using Ollama's native
    format=schema: it's the same underlying model family and stacking bug
    (see structured_chat's docstring - thinking left on can burn the whole
    output budget and return nothing; thinking off can make format get
    silently ignored). Disable thinking, drop format, describe the schema
    in the prompt, parse tolerantly ourselves - identical strategy, now
    also handling an image.

    This also sidesteps the exact failure category the earlier Groq vision
    path hit (json_validate_failed from GROQ'S OWN strict server-side
    schema validation) by construction: there is no server-side strict
    validator here, only _extract_json_object's tolerant parsing - a
    truncated or slightly malformed response becomes an LLMExtractionError
    we control and can retry, not an opaque 400 from a provider.

    image_paths are local filesystem paths - ollama's client reads and
    encodes them itself (no manual base64/data-URI step, unlike the Groq
    vision path's _encode_image_data_uri).
    """
    client = ollama.AsyncClient(host=settings.ollama_host)

    schema_hint = (
        "\n\nRespond with ONLY a single JSON object - no markdown code fences, "
        "no commentary before or after it. Here is the exact shape to return, "
        "with placeholder values showing what kind of content belongs in each "
        "field - replace every placeholder with real content, and do not "
        "return this structure with the placeholders still in it, and do not "
        "return a JSON Schema definition (no 'properties'/'type' keys):\n"
        f"{_build_example_json(schema)}"
    )

    messages = [
        {"role": "system", "content": system_prompt + schema_hint},
        {"role": "user", "content": user_prompt, "images": image_paths},
    ]

    with obs.generation(
        name=f"ollama:{schema.__name__}",
        model=model,
        provider="ollama",
        input=messages,
        model_parameters={
            "temperature": temperature,
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": num_predict or settings.ollama_num_predict,
            "keep_alive": "30m",
            "think": False,
        },
        metadata={"n_images": len(image_paths), "images": [Path(p).name for p in image_paths]},
    ) as gen:
        response = await client.chat(
            model=model,
            messages=messages,
            think=False,
            keep_alive="30m",
            options={
                "temperature": temperature,
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": num_predict or settings.ollama_num_predict,
            },
        )

        usage, meta, done_reason = _ollama_telemetry(response)
        if gen.active:
            meta.update(await obs.ollama_memory(settings.ollama_host, model))

        raw_content = response["message"]["content"] or ""

        if not raw_content.strip():
            # Same leak-into-thinking-field behavior as structured_chat, even
            # with think=False - see that function's docstring.
            raw_content = response["message"].get("thinking") or ""

        gen.record(
            output=raw_content,
            usage=usage,
            metadata=meta,
            warning="done_reason=length: output hit num_predict" if done_reason == "length" else None,
        )

        json_text = _extract_json_object(raw_content)

        try:
            parsed = json.loads(json_text)
            return schema.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001 - we want to wrap any parse/validation error
            raise LLMExtractionError(
                f"Model output failed schema validation: {exc}\nRaw output: {raw_content[:500]!r}"
            ) from exc


async def structured_chat_groq(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    temperature: float = 0.2,
) -> T:
    """
    Call a Groq-hosted model and coerce its response into `schema`.

    Unlike Ollama+qwen3.5 (see structured_chat's docstring), Groq's
    OpenAI-compatible API supports native JSON-schema-constrained decoding
    without the thinking-trace/format collision bug - so this path uses
    response_format directly: the schema is enforced server-side rather than
    just requested in text and parsed tolerantly afterward.

    NOTE: verify against Groq's current docs which models have "strict"
    json_schema support before relying on this for a new model - schema
    enforcement strength has varied by model even within one provider.
    """
    client = AsyncGroq(api_key=settings.groq_api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with obs.generation(
        name=f"groq:{schema.__name__}",
        model=model,
        provider="groq",
        input=messages,
        model_parameters={"temperature": temperature, "max_tokens": settings.groq_max_tokens},
    ) as gen:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=settings.groq_max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001 - network/API errors also need to trigger fallback in nodes.py
            raise LLMExtractionError(f"Groq API call failed for model {model!r}: {exc}") from exc

        raw_content = response.choices[0].message.content or ""

        usage, meta, finish_reason = _groq_telemetry(response)
        gen.record(
            output=raw_content,
            usage=usage,
            metadata=meta,
            warning="finish_reason=length: output hit max_tokens" if finish_reason == "length" else None,
        )

        try:
            parsed = json.loads(raw_content)
            return schema.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            raise LLMExtractionError(
                f"Groq output failed schema validation for model {model!r}: {exc}\nRaw output: {raw_content[:500]!r}"
            ) from exc


async def structured_chat_groq_vision(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[str],
    schema: Type[T],
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> T:
    """
    Vision-capable counterpart to structured_chat_groq, for the Image
    Selection Agent's candidate-ranking calls.

    Same response_format=json_schema mechanism as structured_chat_groq -
    this isn't a different enforcement path, just a different message shape
    (image_url content blocks alongside text in the user message). image_paths
    are local filesystem paths, not remote URLs, since these are files this
    project generated (Agent 4's local_path) - each is read and
    base64-encoded into a data URI rather than passed as a hosted link.

    NOTE: Groq's vision-capable models have historically capped how many
    images fit in a single request, and that cap has changed as the
    available vision models themselves have changed - verify the current
    limit for whichever model `settings.image_selection_model` points at.
    Callers are expected to have already trimmed image_paths to a sane
    count (see nodes.py's use of image_selection_max_images_per_call);
    this function does not enforce a limit itself, since the true limit is
    a property of the model, not of this call.
    """
    client = AsyncGroq(api_key=settings.groq_api_key)

    content: list[dict] = [{"type": "text", "text": user_prompt}]
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_image_data_uri(path)},
            }
        )

    # Trace input is the text only. `content` above carries every image as a
    # base64 data URI - megabytes per call - which must never be shipped to
    # the trace; the image names go in metadata instead.
    traced_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with obs.generation(
        name=f"groq:{schema.__name__}",
        model=model,
        provider="groq",
        input=traced_messages,
        model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        metadata={"n_images": len(image_paths), "images": [Path(p).name for p in image_paths]},
    ) as gen:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                    },
                },
            )
        except RateLimitError as exc:
            # Caught separately from the generic Exception branch below - see
            # LLMRateLimitError's docstring for why this needs to be
            # distinguishable to callers, not just another LLMExtractionError.
            raise LLMRateLimitError(f"Groq vision API rate limit hit for model {model!r}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network/API errors also need to trigger fallback in nodes.py
            raise LLMExtractionError(f"Groq vision API call failed for model {model!r}: {exc}") from exc

        raw_content = response.choices[0].message.content or ""

        usage, meta, finish_reason = _groq_telemetry(response)
        gen.record(
            output=raw_content,
            usage=usage,
            metadata=meta,
            warning="finish_reason=length: output hit max_tokens" if finish_reason == "length" else None,
        )

        try:
            parsed = json.loads(raw_content)
            return schema.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            raise LLMExtractionError(
                f"Groq vision output failed schema validation for model {model!r}: {exc}\nRaw output: {raw_content[:500]!r}"
            ) from exc