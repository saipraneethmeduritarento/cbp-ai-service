"""Provider-agnostic LLM layer — the entire abstraction in one file.

Any code that talks to an LLM in this app goes through here: `get_llm()` / `get_embedder()`
for direct calls, or — far more usually — a ready-made task function near the bottom (e.g.
`summarize_acbp_plan`, `generate_frac_batch`, `embed_search_query`). Callers never import a
provider SDK directly and never build a prompt, response schema or generation config outside
this file.

Sections, top to bottom:
  * errors                  — LLMError / LLMSchemaError
  * neutral data model      — Role, Part, Message, GenerationConfig, Usage, LLMResponse
  * usage logging           — one structured log line per LLM call
  * model-name redirection  — LLM_MODEL_MAP, so a hardcoded model literal can still follow a
                               provider switch without editing the call site
  * the port                — Capability, LLMProvider / EmbeddingProvider ABCs
  * Gemini adapter          — native google-genai (the default)
  * LangChain adapter       — OpenAI / Anthropic / others
  * Langfuse tracing        — a decorator over the port; installed by the factory only when
                               tracing is switched on (see src/core/tracing.py)
  * factory                 — get_llm() / get_embedder(), the one place that picks which
                               adapter backs the port
  * tasks                   — every prompt / response schema / generation config this app
                               uses, one function per LLM-backed feature

`langchain` is an optional dependency (see the `langchain` extra in pyproject.toml). Importing
THIS FILE must never require it — only actually selecting LLM_PROVIDER=langchain should. That
is why `import langchain...` does not appear at module level: LangChainProvider and
LangChainEmbeddingProvider import it lazily inside their own __init__, which only runs when
get_llm()/get_embedder() actually construct one of them.
"""
import asyncio
import base64
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Sequence, Union

from google import genai
from google.genai import types as gtypes
from pydantic import BaseModel, Field

from ..core import tracing
from ..core.configs import LLMProviderOption, settings
from ..core.logger import logger
from ..core.tracing import traced_task
from ..prompts.prompts import (
    ACBP_DOCUMENT_SUMMARY_PROMPT,
    COURSE_SELECTION_SYSTEM_PROMPT,
    DESIGNNATION_GROUP_SYSTEM_PROMPT,
    DOCUMENT_SUMMARY_PROMPT,
    META_SUMMARY_PROMPT,
    VECTOR_QUERY_SYSTEM_PROMPT,
)
from ..prompts.prompts import DESIGNATION_ROLE_MAPPING_PROMPT as DESIGNATION_ROLE_MAPPING_PROMPT_V1
from ..prompts.prompts import ROLE_MAPPING_PROMPT_V2 as ROLE_MAPPING_PROMPT_V1_CENTRE
from ..prompts.prompts import ROLE_MAPPING_PROMPT_V5_STATE as ROLE_MAPPING_PROMPT_V1_STATE
from ..prompts.v2.prompts import DESIGNATION_ROLE_MAPPING_PROMPT as DESIGNATION_ROLE_MAPPING_PROMPT_V2
from ..prompts.v2.prompts import ROLE_MAPPING_PROMPT_V2 as ROLE_MAPPING_PROMPT_V2_CENTRE
from ..prompts.v2.prompts import ROLE_MAPPING_PROMPT_V5_STATE as ROLE_MAPPING_PROMPT_V2_STATE
from ..prompts.v3.prompts import (
    DESIGNATION_EXTRACTION_PROMPT,
    ROLE_MAPPING_PROMPT_CENTRE as ROLE_MAPPING_PROMPT_V3_CENTRE,
    ROLE_MAPPING_PROMPT_STATE as ROLE_MAPPING_PROMPT_V3_STATE,
)
from ..schemas.role_mapping import OrgType


# =============================================================================
# Errors
# =============================================================================

class LLMError(Exception):
    """Base class for all LLM adapter errors."""


class LLMSchemaError(LLMError):
    """A provider did not return JSON matching the requested schema, even after a repair retry."""


# =============================================================================
# Neutral data model — no provider SDK symbols appear here or in the tasks section
# =============================================================================

class Role(str, Enum):
    USER = "user"
    MODEL = "model"


class SafetyPolicy(str, Enum):
    DEFAULT = "default"
    PERMISSIVE = "permissive"            # all four harm categories set to OFF
    HATE_SPEECH_ONLY = "hate_speech_only"  # only HARM_CATEGORY_HATE_SPEECH OFF; the rest stay at
                                           # provider defaults. Used by the public-course lookup,
                                           # which upstream deliberately narrowed to this one
                                           # category rather than disabling all filtering.


class Tool(str, Enum):
    WEB_SEARCH = "web_search"


@dataclass(frozen=True)
class Part:
    text: str | None = None
    data: bytes | None = None
    mime_type: str | None = None

    @classmethod
    def from_text(cls, text: str) -> "Part":
        return cls(text=text)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> "Part":
        return cls(data=data, mime_type=mime_type)


@dataclass
class Message:
    role: Role
    parts: list[Part]

    @classmethod
    def user(cls, *items: Union[str, Part]) -> "Message":
        return cls(
            role=Role.USER,
            parts=[Part.from_text(i) if isinstance(i, str) else i for i in items],
        )


@dataclass
class GenerationConfig:
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    seed: int | None = None
    system_instruction: str | None = None
    json_output: bool = False
    response_schema: Any = None  # a pydantic BaseModel subclass, or a raw JSON-schema dict
    safety: SafetyPolicy = SafetyPolicy.DEFAULT
    thinking_budget: int | None = None  # None=provider default, -1=dynamic, 0=off, N=fixed budget
    include_thoughts: bool = False
    tools: list[Tool] = field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    usage: Usage | None
    model: str
    provider: str
    raw: Any = None


# =============================================================================
# Per-call usage logging (token usage was previously unlogged)
# =============================================================================

@dataclass
class UsageRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def emit_usage(record: UsageRecord) -> None:
    logger.info(
        f"LLM usage: provider={record.provider} model={record.model} "
        f"input_tokens={record.input_tokens} output_tokens={record.output_tokens} "
        f"latency_ms={record.latency_ms:.0f}"
    )


# =============================================================================
# Model-name redirection
#
# A provider switch should be a config change, but several call sites pass a hardcoded Gemini
# model literal (e.g. model="gemini-2.5-pro") instead of reading a setting, so they cannot
# follow LLM_PROVIDER on their own. LLM_MODEL_MAP redirects any model name at the boundary, so
# no call site has to change. An empty map (the default) is pure pass-through.
# =============================================================================

# Recognised LangChain provider prefixes. An allowlist rather than a shape heuristic because
# colons appear inside model names too — OpenAI fine-tunes ("ft:gpt-4o:acme:id") and Bedrock
# ARNs would both be mis-split by a "looks like a token" rule, silently routing to a provider
# named "ft" or "arn". Add an entry here to use a provider LangChain supports but this list
# omits; an unrecognised prefix is treated as part of the model name and warned about.
_KNOWN_PROVIDERS = frozenset({
    "openai", "azure_openai", "anthropic", "google_genai", "google_vertexai", "google_anthropic_vertex",
    "bedrock", "bedrock_converse", "cohere", "fireworks", "together", "mistralai", "groq",
    "ollama", "huggingface", "deepseek", "xai", "perplexity", "nvidia", "ibm", "databricks",
})

# The Google subset of the above — used by BOTH adapters: the Gemini adapter to tell "this
# redirect still points at Google" from "this redirect points at another vendor", and the
# LangChain adapter to pick Google credentials, project/location kwargs, content-block shape
# and native-schema handling. Must remain a subset of _KNOWN_PROVIDERS, since a prefix outside
# that set is never produced by split_provider and so could never match here.
_GOOGLE_PROVIDERS = frozenset({"google_genai", "google_vertexai"})
assert _GOOGLE_PROVIDERS <= _KNOWN_PROVIDERS

_redirect_warned: set[str] = set()
_unknown_prefix_warned: set[str] = set()


def resolve_model(model: str) -> str:
    """Apply LLM_MODEL_MAP to `model`. Returns the mapped name, or `model` unchanged."""
    mapped = (settings.LLM_MODEL_MAP or {}).get(model)
    if not mapped or mapped == model:
        return model
    # Log once per distinct redirection — silent model substitution is the kind of thing that
    # makes "why is this answer different?" take an afternoon to track down.
    if model not in _redirect_warned:
        _redirect_warned.add(model)
        logger.info(f"LLM_MODEL_MAP: routing model '{model}' -> '{mapped}'")
    return mapped


def split_provider(model: str) -> tuple[str | None, str]:
    """Split a LangChain-style "provider:model" string into (provider, model).

    Returns (None, model) when there is no recognised provider prefix — including when the
    colon belongs to the model name itself.
    """
    if ":" not in model:
        return None, model
    prefix, rest = model.split(":", 1)
    if prefix in _KNOWN_PROVIDERS and rest:
        return prefix, rest
    if prefix not in _unknown_prefix_warned:
        _unknown_prefix_warned.add(prefix)
        logger.warning(
            f"Model '{model}' has an unrecognised provider prefix '{prefix}'; treating the whole "
            f"string as a model name. If '{prefix}' is a LangChain provider you intend to use, "
            f"add it to _KNOWN_PROVIDERS in src/services/llm_service.py."
        )
    return None, model


# =============================================================================
# The port: what every adapter implements and all application code depends on
# =============================================================================

class Capability(str, Enum):
    """Optional provider features a task may need. Every member must correspond to something
    a task actually branches on — see LLMProvider.supports()."""
    STREAMING = "streaming"
    NATIVE_JSON_SCHEMA = "native_json_schema"
    NATIVE_FILE_INPUT = "native_file_input"
    WEB_SEARCH = "web_search"
    THINKING = "thinking"


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def supports(self, capability: Capability) -> bool: ...

    @abstractmethod
    async def generate(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]: ...

    async def generate_structured(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        schema: Any,
        config: GenerationConfig | None = None,
    ) -> Any:
        """Generate JSON constrained to `schema` (a pydantic BaseModel subclass, or a raw
        JSON-schema-shaped dict) and parse it. One repair retry on an empty or invalid
        response — this is what lets an adapter with no native schema support still work
        correctly; Gemini constrains natively via response_schema, so the retry path is
        rarely exercised there."""
        base_config = replace(config or GenerationConfig(), json_output=True, response_schema=schema)
        is_model = isinstance(schema, type) and issubclass(schema, BaseModel)

        attempt_contents = contents
        last_error: Exception | None = None
        for attempt in range(2):
            response = await self.generate(attempt_contents, model=model, config=base_config)
            text = response.text
            if not text:
                last_error = ValueError(f"Empty response from {self.name}/{model}")
            else:
                try:
                    data = json.loads(text)
                    return schema.model_validate(data) if is_model else data
                except Exception as e:
                    last_error = e
            if attempt == 0:
                repair = Message.user(
                    f"Your previous output was empty or invalid ({last_error}). "
                    "Return ONLY valid JSON matching the required schema."
                )
                base = [Message.user(contents)] if isinstance(contents, str) else list(contents)
                attempt_contents = base + [repair]

        raise LLMSchemaError(
            f"{self.name}/{model} did not return valid JSON for schema after retry: {last_error}"
        )

    def _record_usage(self, *, model: str, usage: Usage | None, latency_ms: float) -> None:
        emit_usage(UsageRecord(
            provider=self.name,
            model=model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_ms=latency_ms,
        ))


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None = None,
    ) -> list[list[float]]: ...

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None = None,
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Batches large text lists so callers don't hand-roll chunking loops."""
        texts = list(texts)
        results: list[list[float]] = []
        for i in range(0, len(texts), max(1, batch_size)):
            batch = texts[i:i + batch_size]
            results.extend(await self.embed_batch(batch, model=model, dimensions=dimensions))
        return results


# =============================================================================
# Gemini adapter — the only place in this codebase allowed to import `google.genai`.
# Talks Vertex AI for generation and API-key mode for embeddings.
# =============================================================================

def _gemini_model(model: str) -> str:
    """Apply LLM_MODEL_MAP, then strip any LangChain-style provider prefix, which this native
    SDK does not understand. A prefix naming a non-Google provider means the map is pointed at
    another vendor while LLM_PROVIDER is still `gemini` — warn rather than send a model name
    that will 404, since the mismatch is a config error, not a transient failure."""
    resolved = resolve_model(model)
    prefix, bare = split_provider(resolved)
    if prefix and prefix not in _GOOGLE_PROVIDERS:
        logger.warning(
            f"LLM_MODEL_MAP routes '{model}' to '{resolved}' (provider '{prefix}'), but "
            f"LLM_PROVIDER=gemini uses the native Google SDK. Sending '{bare}' to Gemini; set "
            f"LLM_PROVIDER=langchain to actually reach '{prefix}'."
        )
    return bare


_PERMISSIVE_SAFETY_SETTINGS = [
    gtypes.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    gtypes.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    gtypes.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    gtypes.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]

# Narrower than the above on purpose — see SafetyPolicy.HATE_SPEECH_ONLY.
_HATE_SPEECH_ONLY_SAFETY_SETTINGS = [
    gtypes.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
]

_SAFETY_SETTINGS_BY_POLICY = {
    SafetyPolicy.PERMISSIVE: _PERMISSIVE_SAFETY_SETTINGS,
    SafetyPolicy.HATE_SPEECH_ONLY: _HATE_SPEECH_ONLY_SAFETY_SETTINGS,
}


def _build_http_options() -> gtypes.HttpOptions:
    return gtypes.HttpOptions(
        retry_options=gtypes.HttpRetryOptions(
            initial_delay=settings.GEMINI_RETRY_INITIAL_DELAY,
            attempts=settings.GEMINI_RETRY_ATTEMPTS,
            exp_base=settings.GEMINI_RETRY_EXP_BASE,
            http_status_codes=settings.GEMINI_RETRY_HTTP_STATUS_CODES,
        )
    )


def _to_gemini_part(part: Part) -> gtypes.Part:
    if part.data is not None:
        return gtypes.Part.from_bytes(data=part.data, mime_type=part.mime_type or "application/octet-stream")
    return gtypes.Part.from_text(text=part.text or "")


def _to_gemini_contents(contents: str | Sequence[Message]) -> list[gtypes.Content]:
    if isinstance(contents, str):
        return [gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=contents)])]
    return [
        gtypes.Content(
            role="model" if message.role == Role.MODEL else "user",
            parts=[_to_gemini_part(p) for p in message.parts],
        )
        for message in contents
    ]


def _to_gemini_schema(schema: Any) -> Any:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    return schema


def _gemini_extract_usage(response: Any) -> Usage:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return Usage()
    return Usage(
        input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        total_tokens=getattr(meta, "total_token_count", 0) or 0,
    )


def _build_gen_kwargs(config: GenerationConfig) -> dict:
    kwargs: dict = {}
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.max_output_tokens is not None:
        kwargs["max_output_tokens"] = config.max_output_tokens
    if config.seed is not None:
        kwargs["seed"] = config.seed
    if config.system_instruction:
        kwargs["system_instruction"] = config.system_instruction
    if config.json_output or config.response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        if config.response_schema is not None:
            kwargs["response_schema"] = _to_gemini_schema(config.response_schema)
    safety_settings = _SAFETY_SETTINGS_BY_POLICY.get(config.safety)
    if safety_settings:
        kwargs["safety_settings"] = safety_settings
    if config.thinking_budget is not None:
        kwargs["thinking_config"] = gtypes.ThinkingConfig(
            thinking_budget=config.thinking_budget, include_thoughts=config.include_thoughts
        )
    if Tool.WEB_SEARCH in config.tools:
        kwargs["tools"] = [{"google_search": {}}]
    return kwargs


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
        self._client = genai.Client(
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOOGLE_PROJECT_LOCATION_GLOBAL,
            vertexai=settings.GOOGLE_GENAI_USE_VERTEXAI,
            http_options=_build_http_options(),
        )
        logger.info("GeminiProvider initialized (Vertex AI)")

    def supports(self, capability: Capability) -> bool:
        return capability in {
            Capability.STREAMING,
            Capability.NATIVE_JSON_SCHEMA,
            Capability.NATIVE_FILE_INPUT,
            Capability.WEB_SEARCH,
            Capability.THINKING,
        }

    async def generate(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        config = config or GenerationConfig()
        model = _gemini_model(model)
        start = time.monotonic()
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=_to_gemini_contents(contents),
            config=gtypes.GenerateContentConfig(**_build_gen_kwargs(config)),
        )
        latency_ms = (time.monotonic() - start) * 1000
        usage = _gemini_extract_usage(response)
        self._record_usage(model=model, usage=usage, latency_ms=latency_ms)
        return LLMResponse(text=response.text or "", usage=usage, model=model, provider=self.name, raw=response)

    async def stream(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        config = config or GenerationConfig()
        model = _gemini_model(model)
        result = await self._client.aio.models.generate_content_stream(
            model=model,
            contents=_to_gemini_contents(contents),
            config=gtypes.GenerateContentConfig(**_build_gen_kwargs(config)),
        )
        async for chunk in result:
            if chunk.text:
                yield chunk.text


class GeminiEmbeddingProvider(EmbeddingProvider):
    name = "gemini"

    def __init__(self):
        self._client = genai.Client(
            api_key=settings.GOOGLE_API_KEY,
            vertexai=False,
            http_options=_build_http_options(),
        )
        logger.info("GeminiEmbeddingProvider initialized (API key)")

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        config = gtypes.EmbedContentConfig(output_dimensionality=dimensions) if dimensions else None
        # One Content per text — NOT a bare list of strings. The SDK reads a list of strings as
        # a single Content with many parts and returns ONE embedding of their concatenation,
        # which silently collapses a batch into one meaningless vector.
        contents = [gtypes.Content(parts=[gtypes.Part.from_text(text=t)]) for t in texts]
        response = await self._client.aio.models.embed_content(
            model=_gemini_model(model), contents=contents, config=config
        )
        embeddings = [e.values for e in response.embeddings]
        if len(embeddings) != len(contents):
            # Guard the invariant every caller relies on: result[i] corresponds to texts[i].
            # Callers zip() these against their inputs, so a short result misaligns them.
            raise LLMError(
                f"gemini embed_content returned {len(embeddings)} embeddings for "
                f"{len(contents)} inputs; refusing to return a misaligned batch"
            )
        return embeddings


# =============================================================================
# LangChain adapter — the escape hatch for providers beyond Gemini (OpenAI, Anthropic,
# Bedrock, ...) without touching any call site. Optional dependency (see the `langchain`
# extra in pyproject.toml) — `langchain`/`langchain_core` are imported lazily inside the two
# provider classes' __init__, never at module level, so importing this file never requires
# the extra; only actually constructing one of these classes (i.e. LLM_PROVIDER=langchain)
# does.
#
# Google is reached through `langchain-google-genai`, NOT `langchain-google-vertexai`: the
# latter's ChatVertexAI / VertexAIEmbeddings are deprecated as of LangChain 3.2 and slated for
# removal in 4.0. langchain-google-genai serves both the Gemini Developer API and Gemini on
# Vertex AI (`vertexai=True` + project/location), and exposes `output_dimensionality` on
# embeddings, so it covers everything this codebase needs from the Google side.
#
# Model names arriving here may carry a LangChain "provider:model" prefix (e.g.
# "openai:gpt-5"), courtesy of LLM_MODEL_MAP above. A bare name defaults to Google, since
# every call site in this codebase passes a bare Gemini model name.
# =============================================================================

# Provider used when a model name carries no "provider:" prefix. Bare names in this codebase
# are always Gemini models.
# Max concurrent single-text embedding calls (see LangChainEmbeddingProvider.embed_batch).
_EMBED_CONCURRENCY = 8

_DEFAULT_PROVIDER = "google_genai"

# provider prefix -> (settings attribute holding the key, env var the integration reads)
_API_KEY_ENV = {
    "openai": ("OPENAI_API_KEY", "OPENAI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
}

_GEMINI_TYPE_TO_JSON_SCHEMA_TYPE = {
    "OBJECT": "object",
    "ARRAY": "array",
    "STRING": "string",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
}

# Key used to wrap a non-object-root schema (see _normalize_schema_dict) — distinctive enough
# that it will not collide with a real property name in this codebase's schemas.
_WRAPPED_RESULT_KEY = "_wrapped_result"


def _ensure_credentials(provider: str) -> None:
    """Export whatever credential the target integration reads from the environment.

    GeminiProvider does this for Google too, but that constructor never runs when
    LLM_PROVIDER=langchain, so this adapter must set it itself.
    """
    if provider in _GOOGLE_PROVIDERS:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
        return
    entry = _API_KEY_ENV.get(provider)
    if not entry:
        # An unrecognised provider is not necessarily wrong (LangChain supports many more);
        # it just means we have no key wiring for it, so it must come from the ambient env.
        logger.info(
            f"No credential wiring for LangChain provider '{provider}'; relying on the "
            "ambient environment to supply it."
        )
        return
    settings_attr, env_var = entry
    key = getattr(settings, settings_attr, "") or ""
    if key:
        os.environ[env_var] = key
    elif not os.environ.get(env_var):
        logger.warning(
            f"Provider '{provider}' selected but neither {settings_attr} (.env) nor {env_var} "
            "(environment) is set — calls will fail authentication."
        )


def _normalize_schema_dict(schema: dict, *, title: str = "StructuredOutput") -> tuple[dict, bool]:
    """Convert a Gemini-native raw dict schema into standard JSON Schema for providers that
    route structured output through OpenAI-style function calling.

    Two transformations, both required by that path and both harmless to a real JSON Schema:
      * lowercase Gemini's uppercase `type` values ("OBJECT" -> "object", ...)
      * wrap a non-object root in a single-property object, because function-call parameters
        must be an object; several schemas in this codebase are ARRAY-rooted (the
        course-filtering response, DOMAIN_FROM_WAO_SCHEMA), which Gemini's native
        response_schema permits but function calling does not.
    A top-level `title` is also added, since it becomes the synthesized tool name.

    Returns (schema, wrapped); when `wrapped` is True the caller must unwrap
    `_WRAPPED_RESULT_KEY` from the parsed result.
    """
    def _convert(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        converted: dict = {}
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                converted[key] = _GEMINI_TYPE_TO_JSON_SCHEMA_TYPE.get(value, value.lower())
            elif key == "properties" and isinstance(value, dict):
                converted[key] = {name: _convert(prop) for name, prop in value.items()}
            elif key == "items":
                converted[key] = _convert(value)
            else:
                converted[key] = value
        return converted

    converted = _convert(schema)
    wrapped = converted.get("type") != "object"
    if wrapped:
        converted = {
            "type": "object",
            "properties": {_WRAPPED_RESULT_KEY: converted},
            "required": [_WRAPPED_RESULT_KEY],
        }
    converted.setdefault("title", title)
    return converted, wrapped


def _resolve_target(model: str) -> tuple[str, str, dict]:
    """Map a caller-supplied model name to (provider, langchain_model_id, provider_kwargs).

    Google gets project/location passed explicitly rather than relying on the integration's
    own default (observed to be us-central1, not this app's configured
    GOOOGLE_PROJECT_LOCATION_GLOBAL="global") — a model published in one Vertex AI location is
    not necessarily published in another, as gemini-embedding-2 (global-only) demonstrated
    live. Those kwargs are Google-specific and must not leak to other providers, which would
    reject them as unexpected arguments.
    """
    resolved = resolve_model(model)
    provider, bare = split_provider(resolved)
    provider = provider or _DEFAULT_PROVIDER

    kwargs: dict = {}
    if provider in _GOOGLE_PROVIDERS:
        kwargs = {
            "project": settings.GOOGLE_PROJECT_ID,
            "location": settings.GOOOGLE_PROJECT_LOCATION_GLOBAL,
            "vertexai": settings.GOOGLE_GENAI_USE_VERTEXAI,
        }
    return provider, f"{provider}:{bare}", kwargs


def _to_langchain_content_block(part: Part, provider: str) -> dict:
    """Translate a neutral Part into the target provider's content-block shape.

    Binary attachments are the one genuinely non-portable piece of the message format: each
    vendor names and nests them differently, so there is no single block that works everywhere.
    """
    if part.data is None:
        return {"type": "text", "text": part.text or ""}

    mime = part.mime_type or "application/octet-stream"
    encoded = base64.b64encode(part.data).decode("ascii")

    if provider in _GOOGLE_PROVIDERS:
        return {"type": "media", "mime_type": mime, "data": encoded}
    if provider == "anthropic":
        block_type = "image" if mime.startswith("image/") else "document"
        return {
            "type": block_type,
            "source": {"type": "base64", "media_type": mime, "data": encoded},
        }
    if provider == "openai":
        if mime.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
        return {"type": "file", "file": {"filename": "attachment", "file_data": f"data:{mime};base64,{encoded}"}}

    raise ValueError(
        f"LangChainProvider does not know how to attach '{mime}' content for provider "
        f"'{provider}'. Add a content-block mapping in _to_langchain_content_block, or route "
        f"this call to a provider that has one ({sorted(_GOOGLE_PROVIDERS | {'openai', 'anthropic'})})."
    )


def _to_langchain_messages(
    contents: str | Sequence[Message], system_instruction: str | None, provider: str
):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [SystemMessage(content=system_instruction)] if system_instruction else []
    if isinstance(contents, str):
        messages.append(HumanMessage(content=contents))
        return messages
    for message in contents:
        content = [_to_langchain_content_block(p, provider) for p in message.parts]
        cls = AIMessage if message.role == Role.MODEL else HumanMessage
        messages.append(cls(content=content))
    return messages


def _langchain_extract_usage(response: Any) -> Usage:
    meta = getattr(response, "usage_metadata", None) if response is not None else None
    if not meta:
        return Usage()
    return Usage(
        input_tokens=meta.get("input_tokens", 0) or 0,
        output_tokens=meta.get("output_tokens", 0) or 0,
        total_tokens=meta.get("total_tokens", 0) or 0,
    )


class LangChainProvider(LLMProvider):
    name = "langchain"

    def __init__(self):
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as e:
            raise ImportError(
                "LangChainProvider requires the 'langchain' extra: install with "
                "`uv sync --extra langchain` (or `pip install cbp-ai-service[langchain]`)."
            ) from e
        self._init_chat_model = init_chat_model
        self._models: dict[tuple, Any] = {}
        logger.info("LangChainProvider initialized")

    def _get_chat_model(self, model: str, config: GenerationConfig) -> tuple[Any, str, str]:
        """Returns (chat_model, provider, model_id). `model_id` is the fully-resolved
        "provider:model" that actually serves the request — report that rather than the name
        the caller asked for, so a LLM_MODEL_MAP redirect is visible in usage logs instead of
        logs claiming a model that never ran."""
        provider, model_id, provider_kwargs = _resolve_target(model)
        key = (model_id, config.temperature, config.top_p, config.max_output_tokens)
        chat = self._models.get(key)
        if chat is None:
            _ensure_credentials(provider)
            kwargs: dict = {}
            if config.temperature is not None:
                kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if config.max_output_tokens is not None:
                kwargs["max_tokens"] = config.max_output_tokens
            chat = self._init_chat_model(model_id, **kwargs, **provider_kwargs)
            self._models[key] = chat
        return chat, provider, model_id

    def supports(self, capability: Capability) -> bool:
        return capability in {Capability.STREAMING, Capability.NATIVE_JSON_SCHEMA, Capability.NATIVE_FILE_INPUT}

    async def generate(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        config = config or GenerationConfig()
        chat, provider, model_id = self._get_chat_model(model, config)
        messages = _to_langchain_messages(contents, config.system_instruction, provider)

        start = time.monotonic()
        if config.response_schema is not None:
            schema = config.response_schema
            wrapped = False
            is_pydantic = isinstance(schema, type) and issubclass(schema, BaseModel)
            if not is_pydantic:
                # langchain-google-genai forwards a raw dict straight to Gemini's native
                # response_schema, which accepts uppercase types and a non-object root — so
                # leave Google's schemas untouched and only normalize for the
                # function-calling-based providers that actually need it.
                if provider in _GOOGLE_PROVIDERS:
                    schema = dict(schema)
                    schema.setdefault("title", "StructuredOutput")
                else:
                    schema, wrapped = _normalize_schema_dict(schema)
            structured = chat.with_structured_output(schema, include_raw=True)
            result = await structured.ainvoke(messages)
            latency_ms = (time.monotonic() - start) * 1000
            parsed, raw = result.get("parsed"), result.get("raw")
            if wrapped and isinstance(parsed, dict) and _WRAPPED_RESULT_KEY in parsed:
                parsed = parsed[_WRAPPED_RESULT_KEY]
            if isinstance(parsed, BaseModel):
                text = parsed.model_dump_json()
            elif parsed is not None:
                text = json.dumps(parsed)
            else:
                text = getattr(raw, "content", "") or ""
            usage = _langchain_extract_usage(raw)
            self._record_usage(model=model_id, usage=usage, latency_ms=latency_ms)
            return LLMResponse(text=text, usage=usage, model=model_id, provider=self.name, raw=raw)

        response = await chat.ainvoke(messages)
        latency_ms = (time.monotonic() - start) * 1000
        usage = _langchain_extract_usage(response)
        self._record_usage(model=model_id, usage=usage, latency_ms=latency_ms)
        return LLMResponse(text=response.content or "", usage=usage, model=model_id, provider=self.name, raw=response)

    async def stream(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        config: GenerationConfig | None = None,
    ):
        config = config or GenerationConfig()
        chat, provider, _ = self._get_chat_model(model, config)
        messages = _to_langchain_messages(contents, config.system_instruction, provider)
        async for chunk in chat.astream(messages):
            if chunk.content:
                yield chunk.content


class LangChainEmbeddingProvider(EmbeddingProvider):
    name = "langchain"

    def __init__(self):
        try:
            from langchain.embeddings import init_embeddings
        except ImportError as e:
            raise ImportError(
                "LangChainEmbeddingProvider requires the 'langchain' extra: install with "
                "`uv sync --extra langchain` (or `pip install cbp-ai-service[langchain]`)."
            ) from e
        self._init_embeddings = init_embeddings
        self._embedders: dict[tuple, Any] = {}
        logger.info("LangChainEmbeddingProvider initialized")

    def _get_embedder(self, model: str, dimensions: int | None):
        provider, model_id, provider_kwargs = _resolve_target(model)
        # Output width is fixed at construction, not per call, so it belongs in the cache key —
        # otherwise a caller wanting 768 would be handed an embedder built for 1536 (this app
        # legitimately uses both: 768 for designation matching, 1536 for course search).
        key = (model_id, dimensions)
        embedder = self._embedders.get(key)
        if embedder is None:
            _ensure_credentials(provider)
            kwargs = dict(provider_kwargs)
            if dimensions:
                # langchain-google-genai names this output_dimensionality (matching the native
                # SDK); OpenAI's integration calls it dimensions.
                kwargs["output_dimensionality" if provider in _GOOGLE_PROVIDERS else "dimensions"] = dimensions
            embedder = self._init_embeddings(model_id, **kwargs)
            self._embedders[key] = embedder
        return embedder

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        """One embedding per input text, in input order.

        Texts are embedded one call at a time (concurrently), because Gemini's embedContent
        endpoint on the Vertex backend rejects multi-content requests outright:
        "The embedContent API for this model only supports one content at a time." Each call
        still goes through aembed_documents([t]) rather than aembed_query(t) so the task type —
        and therefore the vector space — is identical to a batch call, which matters because
        these vectors are compared against stored ones.
        """
        texts = list(texts)
        if not texts:
            return []
        embedder = self._get_embedder(model, dimensions)
        if len(texts) == 1:
            return await embedder.aembed_documents(texts)

        # Bounded concurrency: a designation-matching request can carry ~100 names, and firing
        # them all at once invites rate limiting.
        semaphore = asyncio.Semaphore(_EMBED_CONCURRENCY)

        async def one(text: str) -> list[float]:
            async with semaphore:
                return (await embedder.aembed_documents([text]))[0]

        return await asyncio.gather(*(one(t) for t in texts))


# =============================================================================
# Langfuse tracing — a decorator over the port, so neither adapter (nor any call site) knows
# tracing exists. Installed by the factories below and only when tracing is actually on: with
# LANGFUSE_ENABLED unset, get_llm() returns the bare adapter and none of this code runs.
#
# What lands in a trace per call: the prompt (text truncated, binary parts replaced by a
# "<application/pdf, 2.4 MB>" placeholder), the generation config, the response text, token
# usage, latency, and ERROR level plus the exception on failure.
# =============================================================================

def _format_bytes(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} bytes"


def _trace_part(part: Part) -> dict:
    """One neutral Part as trace-safe JSON. Binary data is described, never included — a 3 MB
    PDF base64'd into a trace would be both useless and larger than the ingestion limit."""
    if part.data is not None:
        return {"media": f"{part.mime_type or 'application/octet-stream'}, {_format_bytes(len(part.data))}"}
    return {"text": tracing.truncate(part.text or "")}


def _trace_input(contents: str | Sequence[Message]) -> Any:
    if isinstance(contents, str):
        return tracing.truncate(contents)
    return [
        {"role": message.role.value, "parts": [_trace_part(p) for p in message.parts]}
        for message in contents
    ]


def _trace_model_params(config: GenerationConfig) -> dict:
    """Only the parameters actually set — an unset value is the provider default, and recording
    it as null would misreport it as an explicit choice."""
    params = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_output_tokens": config.max_output_tokens,
        "seed": config.seed,
        "thinking_budget": config.thinking_budget,
    }
    return {k: v for k, v in params.items() if v is not None}


def _schema_name(schema: Any) -> str | None:
    if schema is None:
        return None
    if isinstance(schema, type):
        return schema.__name__
    if isinstance(schema, dict):
        return schema.get("title")
    return None


def _trace_metadata(config: GenerationConfig) -> dict:
    metadata: dict = {"json_output": config.json_output, "safety": config.safety.value}
    if config.system_instruction:
        metadata["system_instruction"] = tracing.truncate(config.system_instruction)
    if config.tools:
        metadata["tools"] = [t.value for t in config.tools]
    if (schema_name := _schema_name(config.response_schema)) is not None:
        metadata["response_schema"] = schema_name
    return metadata


def _trace_structured_output(value: Any) -> Any:
    """Parsed structured output for a trace: kept as JSON so Langfuse renders it as a tree,
    unless it is over the payload cap, in which case it degrades to a truncated string."""
    try:
        serialized = json.dumps(value, default=str)
    except Exception:
        return tracing.truncate(str(value))
    if len(serialized) <= settings.LANGFUSE_MAX_PAYLOAD_CHARS:
        return value
    return tracing.truncate(serialized)


def _usage_details(usage: Usage | None) -> dict | None:
    if usage is None:
        return None
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "total": usage.total_tokens,
    }


class TracedLLMProvider(LLMProvider):
    """Wraps an LLMProvider so every call becomes a Langfuse observation.

    Two details worth preserving:
      * `self.name` mirrors the inner adapter's name, because it appears in errors callers see
        ("Empty response from {self.name}/{model}", LLMSchemaError) — wrapping must not change
        those strings.
      * `_record_usage` is left entirely to the inner adapter, so the existing `LLM usage: ...`
        log line is identical whether tracing is on or off. Langfuse is an addition to that
        logging, not a replacement for it.

    `generate_structured` is inherited-and-wrapped rather than delegated: the ABC implementation
    calls `self.generate`, so the first attempt and the repair retry each become a generation
    under one span. If an adapter ever overrides `generate_structured`, this wrapper must be
    changed to delegate to the inner method instead, or that override will be bypassed.
    """

    def __init__(self, inner: LLMProvider):
        self._inner = inner
        self.name = inner.name

    def supports(self, capability: Capability) -> bool:
        return self._inner.supports(capability)

    async def generate(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        config = config or GenerationConfig()
        with tracing.observation(
            f"{self.name}.generate",
            as_type="generation",
            input=_trace_input(contents),
            model=resolve_model(model),
            model_parameters=_trace_model_params(config),
            metadata=_trace_metadata(config),
        ) as observation:
            try:
                response = await self._inner.generate(contents, model=model, config=config)
            except Exception as e:
                observation.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
            observation.update(
                output=tracing.truncate(response.text),
                # The response reports the model that actually served the request, which an
                # LLM_MODEL_MAP redirect can make different from the one asked for.
                model=response.model,
                usage_details=_usage_details(response.usage),
            )
            return response

    async def stream(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        config = config or GenerationConfig()
        # detached_observation, not observation: this is an async generator, and an attached OTEL
        # context cannot be safely carried across `yield`.
        with tracing.detached_observation(
            f"{self.name}.stream",
            as_type="generation",
            input=_trace_input(contents),
            model=resolve_model(model),
            model_parameters=_trace_model_params(config),
            metadata=_trace_metadata(config),
        ) as observation:
            chunks: list[str] = []
            first_chunk_at: datetime | None = None
            try:
                async for chunk in self._inner.stream(contents, model=model, config=config):
                    if first_chunk_at is None:
                        first_chunk_at = datetime.now(timezone.utc)
                    chunks.append(chunk)
                    yield chunk
            except Exception as e:
                observation.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
            finally:
                # In `finally` so a consumer that abandons the stream still gets whatever was
                # produced recorded, rather than an observation with no output at all.
                observation.update(
                    output=tracing.truncate("".join(chunks)),
                    completion_start_time=first_chunk_at,
                )
        # No usage_details: neither adapter's stream() reads token counts off the stream.

    async def generate_structured(
        self,
        contents: str | Sequence[Message],
        *,
        model: str,
        schema: Any,
        config: GenerationConfig | None = None,
    ) -> Any:
        with tracing.observation(
            f"{self.name}.generate_structured",
            as_type="span",
            metadata={"response_schema": _schema_name(schema), "model": resolve_model(model)},
        ) as observation:
            try:
                result = await super().generate_structured(
                    contents, model=model, schema=schema, config=config
                )
            except Exception as e:
                observation.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
            output = result.model_dump() if isinstance(result, BaseModel) else result
            observation.update(output=_trace_structured_output(output))
            return result


class TracedEmbeddingProvider(EmbeddingProvider):
    """Wraps an EmbeddingProvider so each embed_batch — one API request — becomes an
    "embedding" observation.

    The vectors are not recorded: they are hundreds of floats per text, would dwarf every other
    payload in the trace and are unreadable anyway. The count and width are what actually help
    when a batch comes back misaligned or at the wrong dimensionality.

    `embed()` is inherited from the ABC, so a large call that chunks into several batches
    produces one observation per batch.
    """

    def __init__(self, inner: EmbeddingProvider):
        self._inner = inner
        self.name = inner.name

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        texts = list(texts)
        with tracing.observation(
            f"{self.name}.embed_batch",
            as_type="embedding",
            input=[tracing.truncate(t) for t in texts],
            model=resolve_model(model),
            model_parameters={"dimensions": dimensions} if dimensions else {},
        ) as observation:
            try:
                vectors = await self._inner.embed_batch(texts, model=model, dimensions=dimensions)
            except Exception as e:
                observation.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
            observation.update(output={
                "embeddings": len(vectors),
                "dimensions": len(vectors[0]) if vectors else 0,
            })
            return vectors


# =============================================================================
# Factory — the one place that decides which adapter backs the port.
# =============================================================================

@lru_cache(maxsize=None)
def get_llm(provider: LLMProviderOption | None = None) -> LLMProvider:
    provider = provider or settings.LLM_PROVIDER
    if provider == LLMProviderOption.GEMINI:
        return _traced(GeminiProvider())
    if provider == LLMProviderOption.LANGCHAIN:
        return _traced(LangChainProvider())
    raise ValueError(f"Unknown LLM provider: {provider}")


@lru_cache(maxsize=None)
def get_embedder(provider: LLMProviderOption | None = None) -> EmbeddingProvider:
    provider = provider or settings.LLM_EMBEDDING_PROVIDER
    if provider == LLMProviderOption.GEMINI:
        return _traced_embedder(GeminiEmbeddingProvider())
    if provider == LLMProviderOption.LANGCHAIN:
        return _traced_embedder(LangChainEmbeddingProvider())
    raise ValueError(f"Unknown embedding provider: {provider}")


def _traced(provider: LLMProvider) -> LLMProvider:
    """Returns `provider` untouched when tracing is off — no wrapper, no payload serialization."""
    return TracedLLMProvider(provider) if tracing.tracing_enabled() else provider


def _traced_embedder(provider: EmbeddingProvider) -> EmbeddingProvider:
    if tracing.tracing_enabled() and settings.LANGFUSE_TRACE_EMBEDDINGS:
        return TracedEmbeddingProvider(provider)
    return provider


# =============================================================================
# Tasks — every prompt / response schema / generation config this app uses.
#
# Each function below owns exactly three things for one task: the prompt, the response
# schema, and the generation config. It then calls get_llm()/get_embedder() above, so every
# task works identically under LLM_PROVIDER=gemini or =langchain.
#
# Deliberately excluded, so these stay pure and provider-agnostic:
#   * database access  — callers fetch data and pass it in
#   * file/storage I/O — callers read bytes and pass them in
#   * post-processing that is business logic rather than prompting (KCM canonicalization,
#     competency quota enforcement, Redis caching of embeddings) stays in the domain services
#
# Callers are thin: a route or service fetches its data, calls one function here, and handles
# the result. Nothing outside this file builds an LLM prompt or a generation config.
# =============================================================================

# The KCM competency master, loaded here because its only job is to fill {kcm_competencies}
# in the role-mapping prompt templates. v3 uses the id-bearing file (its prompt asks the model
# to return competency_id); v1/v2 use the original. The v3 role-mapping service loads the same
# id-bearing file separately for KCM canonicalization, which is business logic rather than
# prompting and therefore deliberately does not share this copy.
with open("data/competencies.json") as _f:
    _KCM_COMPETENCIES = json.load(_f)

with open("data/withidentifier_competencies.json") as _f:
    _KCM_COMPETENCIES_V3 = json.load(_f)


def _is_state(organization_data: Dict[str, Any]) -> bool:
    """Which role-mapping prompt variant to use. v2/v3 key off org_type; v1 keys off whether a
    department is set (see _build_role_mapping_prompt_v1) — the two rules are not the same."""
    return organization_data.get("org_type") == OrgType.state.value

# Model used for the v3 WAO/domain pass and several legacy call sites that pinned a literal
# rather than reading a setting. Kept as a literal so behaviour is unchanged; redirect it via
# LLM_MODEL_MAP rather than editing here.
LEGACY_PRO_MODEL = "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class Designation(BaseModel):
    sort_order: int = Field(description="Hierarchical position, starting from 1 (highest) and incrementing sequentially")
    designation: str = Field(description="Exact official designation or job title")
    wing_division_section: str = Field(description="Exact wing/division/section the designation belongs to, or org unit / administrative from source document")


class DesignationExtractionResponse(BaseModel):
    designations: List[Designation] = Field(description="List of extracted unique designations sorted by hierarchy")


class FRACCompetency(BaseModel):
    competency_id: Optional[str] = Field(default=None, description="KCM competency id (e.g. BEH-07 / FUN-23). REQUIRED for Behavioural & Functional; omit for Domain.")
    type: Literal["Behavioural", "Functional", "Domain"] = Field(description="Competency type: Behavioural, Functional, or Domain")
    theme: str = Field(description="Competency theme")
    sub_theme: str = Field(description="Competency sub theme")


class FRACRoleMapping(BaseModel):
    designation_name: str = Field(description="Official designation name")
    wing_division_section: str = Field(description="Wing/division/section the designation belongs to")
    role_responsibilities: List[str] = Field(description="Flat list of role responsibilities as strings")
    activities: List[str] = Field(description="Flat list of activity strings")
    sort_order: int = Field(description="Hierarchy sort order, strictly increasing from 1")
    competencies: List[FRACCompetency] = Field(description="Flat list of competency objects.")
    source: Optional[List[str]] = Field(default=None, description="Source references")


class FRACBatchResponse(BaseModel):
    mappings: List[FRACRoleMapping] = Field(description="List of FRAC role mappings for all designations in the batch")


# Raw dict schemas. These use Gemini's native uppercase type names; the LangChain adapter
# normalizes them automatically for providers that need standard JSON Schema.
_DESIGNATION_ROLE_MAPPING_SCHEMA_V1 = {"type":"OBJECT","properties":{"designation_name":{"type":"STRING","description":"The official designation or job title for the role."},"wing_division_section":{"type":"STRING","description":"The organizational unit (wing, division, or section) where the role is situated."},"role_responsibilities":{"type":"ARRAY","items":{"type":"STRING"},"description":"A list of 5-8 concise, action-oriented role responsibilities."},"activities":{"type":"ARRAY","items":{"type":"STRING"},"description":"A list of 5–8 activities or tasks aligned to the role responsibilities."},"competencies":{"type":"ARRAY","items":{"type":"OBJECT","properties":{"type":{"type":"STRING","enum":["Behavioral","Functional","Domain"],"description":"The category of competency as per Karmayogi framework."},"theme":{"type":"STRING","description":"The parent theme of the competency (must come from dataset)."},"sub_theme":{"type":"STRING","description":"The sub-theme of the competency (must come from dataset)."}},"required":["type","theme","sub_theme"]},"description":"A list of competencies relevant to the role. Must include at least one Behavioral, one Functional, and one Domain competency."}},"required":["designation_name","wing_division_section","role_responsibilities","activities","competencies"]}

# v2 differs from v1 only in the British "Behavioural" spelling of the competency type enum.
_DESIGNATION_ROLE_MAPPING_SCHEMA_V2 = {"type":"OBJECT","properties":{"designation_name":{"type":"STRING","description":"The official designation or job title for the role."},"wing_division_section":{"type":"STRING","description":"The organizational unit (wing, division, or section) where the role is situated."},"role_responsibilities":{"type":"ARRAY","items":{"type":"STRING"},"description":"A list of 5-8 concise, action-oriented role responsibilities."},"activities":{"type":"ARRAY","items":{"type":"STRING"},"description":"A list of 5–8 activities or tasks aligned to the role responsibilities."},"competencies":{"type":"ARRAY","items":{"type":"OBJECT","properties":{"type":{"type":"STRING","enum":["Behavioural","Functional","Domain"],"description":"The category of competency as per Karmayogi framework."},"theme":{"type":"STRING","description":"The parent theme of the competency (must come from dataset)."},"sub_theme":{"type":"STRING","description":"The sub-theme of the competency (must come from dataset)."}},"required":["type","theme","sub_theme"]},"description":"A list of competencies relevant to the role. Must include at least one Behavioural, one Functional, and one Domain competency."}},"required":["designation_name","wing_division_section","role_responsibilities","activities","competencies"]}

_CONTEXTUAL_QUERIES_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "keyword_query":     {"type": "STRING"},
        "description_query": {"type": "STRING"},
        "combined_query":    {"type": "STRING"},
        "search_keywords":   {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["keyword_query", "description_query", "combined_query", "search_keywords"],
}

_DESIGNATION_GROUP_SCHEMA = {
    "type": "OBJECT",
    "properties": {"group": {"type": "STRING", "enum": ["AB", "CD"]}},
    "required": ["group"],
}

_COURSE_FILTER_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "identifier": {"type": "STRING"},
            "course":     {"type": "STRING"},
            "relevancy":  {"type": "INTEGER"},
            "rationale":  {"type": "STRING"},
        },
        "required": ["identifier", "course", "relevancy", "rationale"],
    },
}


# ---------------------------------------------------------------------------
# Prompt output-format scaffolding (illustrative JSON shapes injected into prompts)
# ---------------------------------------------------------------------------

_V1_CENTRE_OUTPUT = [{
    "designation_name": "string", "wing_division_section": "string",
    "role_responsibilities": ["string", "string"], "activities": ["string", "string"],
    "sort_order": "integer",
    "competencies": [{"type": "Behavioral | Functional | Domain", "theme": "string", "sub_theme": "string", "source": "KCM or AI Suggested"}],
    "source": ["ACBP", "Work Allocation Order", "AI Suggested"],
}]

_V1_STATE_OUTPUT = [{
    "designation_name": "string", "wing_division_section": "string",
    "role_responsibilities": ["string", "string"], "activities": ["string", "string"],
    "sort_order": "integer",
    "competencies": [{"type": "Behavioral | Functional | Domain", "theme": "string", "sub_theme": "string", "source": "KCM or AI Suggested"}],
    # `or` between string literals is almost certainly a typo for a comma in the original v1
    # service — Python collapses it to just "Work Allocation Order". Preserved verbatim so the
    # v1 prompt is byte-identical to before this refactor; fixing it would change v1 output.
    "source": ["Work Allocation Order" or "ACBP" or "Additional supporting document" or "AI Suggested"],
}]

_V2_OUTPUT = [{
    "designation_name": "string", "wing_division_section": "string",
    "role_responsibilities": ["string", "string"], "activities": ["string", "string"],
    "sort_order": "integer",
    "competencies": [{"type": "Behavioral | Functional | Domain", "theme": "string", "sub_theme": "string", "source": "KCM or AI Suggested"}],
    "source": ["Primary document summaries", "AI Suggested"],
}]

_V3_MAPPING_TEMPLATE = {
    "designation_name": "string", "wing_division_section": "string",
    "role_responsibilities": ["string", "string"], "activities": ["string", "string"],
    "sort_order": "integer",
    "competencies": [{
        "competency_id": "KCM id e.g. BEH-07 / FUN-23 (REQUIRED for Behavioural & Functional; omit for Domain)",
        "type": "Behavioural | Functional | Domain", "theme": "string", "sub_theme": "string",
    }],
}
_V3_CENTRE_OUTPUT = {"mappings": [{**_V3_MAPPING_TEMPLATE, "source": ["ACBP", "Work Allocation Order", "AI Suggested"]}]}
_V3_STATE_OUTPUT = {"mappings": [{**_V3_MAPPING_TEMPLATE, "source": ["Work Allocation Order", "ACBP", "Additional supporting document", "AI Suggested"]}]}

_WORK_ALLOCATION_SUMMARY_PROMPT = """
You are an expert analyst specializing in Work Allocation Order documents for government and organizational projects. Please analyze the provided Work Allocation Order PDF document and create a comprehensive, structured summary.

Focus on extracting and summarizing:
1. Work order details and reference numbers
2. Allocated tasks and responsibilities
3. Personnel assignments and roles
4. Timeline and deadlines
5. Resource allocation and budget
6. Deliverables and expected outcomes
7. Quality standards and requirements
8. Reporting and monitoring mechanisms
9. Approval authorities and stakeholders

**Output Format:**
Provide a well-structured summary that captures all essential elements of the Work Allocation Order. Use clear headings, bullet points where appropriate, and ensure the summary provides actionable insights for project execution.

Please analyze the PDF document and provide your comprehensive summary:
"""

_GENERAL_COURSES_SYSTEM_PROMPT = """
        You are an expert in civil service training and development.
        Your role is to recommend highly relevant and foundational courses that would help professionals excel in their designation within government/administrative organizations.

        # Research & Recommendation Guidelines:
        1. Search across credible and accessible learning platforms, including but not limited to:
            Coursera, edX, Udemy, FutureLearn, SWAYAM, NPTEL, Khan Academy, WHO, Harvard Online, MIT OCW, Stanford Online, LinkedIn Learning, etc.
            - Prefer globally credible and India-contextualized content.
            - Do not include iGOT/Karmayogi links.

        2. Course Selection Criteria:
            - Recommend 10–15 courses that are universally essential for this designation.
            - Courses must strengthen Behavioral, Functional, and Domain competencies.
            - Ensure recommendations are active, course-specific, and not generic category pages.
            - Do not include fictional or AI-generated course names. Recommend only courses that exist publicly and are accessible.

        3. Quality Control:
            - Avoid duplicates.
            - Ensure public links are correct and accessible.
            - Keep rationales concise and role-relevant.
            - Course name should be the same as given in the webpage.

        For each course, provide the following information in a structured JSON format:
        - course: The full name of the course.
        - platform: The name of the platform where the course is hosted (e.g., Coursera, edX, Udemy).
        - relevancy: An integer from 0 to 100, indicating high relevancy.
        - rationale: A brief, 1-2 sentence explanation of why this course is essential.
        - language: The language of the specific course (e.g., en, hi).
        - public_link: An actual public URL to the specific course.
        - competencies: An array of competency objects.
          Each object should have competencyAreaName, competencyThemeName, and competencySubThemeName.
        Ensure the output is a JSON array of objects.

        **OUTPUT FORMAT REQUIRED:**
        Provide the output as a **direct JSON array of objects**.
        **IMPORTANT:** Do **NOT** enclose the JSON within markdown code blocks (e.g., do not use ```json ... ``` or ``` ... ```). The output must be *only* the JSON array itself.
        """


def _strip_code_fences(text: str) -> str:
    """Several legacy prompts ask for bare JSON but the model sometimes fences it anyway."""
    return text.replace("```json", "").replace("```", "")


# ---------------------------------------------------------------------------
# Document & PDF summaries
# ---------------------------------------------------------------------------

async def _summarize_pdf(pdf_bytes: bytes, prompt: str, label: str) -> str:
    """Native PDF understanding: the raw bytes go to the model, so scanned pages, tables and
    multi-column layouts are handled without a text-extraction step."""
    contents = [Message.user(Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt)]
    config = GenerationConfig(
        temperature=1, top_p=0.95, seed=0, max_output_tokens=65535,
        safety=SafetyPolicy.PERMISSIVE, thinking_budget=-1,
    )
    response = await get_llm().generate(contents, model=LEGACY_PRO_MODEL, config=config)
    if not response.text:
        logger.warning(f"Empty {label} summary generated")
        return ""
    logger.info(f"Successfully generated {label} summary ({len(response.text)} characters)")
    return response.text


@traced_task()
async def summarize_acbp_plan(pdf_bytes: bytes) -> str:
    """Summarize an ACBP Plan PDF. Returns "" on an empty model response."""
    logger.info("Generating ACBP Plan summary")
    return await _summarize_pdf(pdf_bytes, ACBP_DOCUMENT_SUMMARY_PROMPT, "ACBP Plan")


@traced_task()
async def summarize_work_allocation(pdf_bytes: bytes) -> str:
    """Summarize a Work Allocation Order PDF. Returns "" on an empty model response."""
    logger.info("Generating Work Allocation Order summary")
    return await _summarize_pdf(pdf_bytes, _WORK_ALLOCATION_SUMMARY_PROMPT, "Work Allocation Order")


@traced_task()
async def summarize_uploaded_document(pdf_bytes: bytes) -> str:
    """Summarize an uploaded document PDF (the /files/{id}/summary background task)."""
    contents = [Message.user(Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), DOCUMENT_SUMMARY_PROMPT)]
    config = GenerationConfig(temperature=0, top_p=1, safety=SafetyPolicy.PERMISSIVE)
    response = await get_llm().generate(contents, model=settings.GEMINI_PRO_MODEL_NAME, config=config)
    return response.text


@traced_task()
async def summarize_across_documents(joined_summaries: str) -> str:
    """Roll several per-document summaries up into one meta-summary."""
    contents = [Message.user(META_SUMMARY_PROMPT.format(payload=joined_summaries))]
    config = GenerationConfig(
        temperature=0.6, top_p=0.95, max_output_tokens=8192, safety=SafetyPolicy.PERMISSIVE,
    )
    response = await get_llm().generate(contents, model=LEGACY_PRO_MODEL, config=config)
    return response.text


# ---------------------------------------------------------------------------
# Role mapping — v1
# ---------------------------------------------------------------------------

def _build_role_mapping_prompt_v1(organization_data: Dict[str, Any]) -> str:
    is_state = bool(organization_data["department_id"])
    logger.info(f"Role Mapping is using prompt :: {'STATE_PROMPT' if is_state else 'CENTER_PROMPT'}")
    prompt = ROLE_MAPPING_PROMPT_V1_STATE if is_state else ROLE_MAPPING_PROMPT_V1_CENTRE
    output_format = _V1_STATE_OUTPUT if is_state else _V1_CENTRE_OUTPUT
    return prompt.format(
        organization_name=organization_data.get("organization_name"),
        department_name=organization_data.get("department_name"),
        sector=organization_data.get("sector"),
        instructions=organization_data.get("instruction"),
        acbp_summary=organization_data.get("acbp_plan_summary"),
        work_allocation_summary=organization_data.get("work_allocation_summary"),
        kcm_competencies=json.dumps(_KCM_COMPETENCIES, indent=2),
        output_json_format=json.dumps(output_format, indent=2),
    )


@traced_task()
async def generate_role_mapping_v1(
    organization_data: Dict[str, Any],
    additional_document_contents: List[bytes] | None = None,
) -> Any:
    """v1 single-pass role mapping. Returns the parsed JSON (a list of mappings).

    This prompt asks for JSON in prose rather than using a response schema, so the output is
    fence-stripped and json.loads'd rather than schema-validated — preserved as-is.
    """
    logger.info(f"Generating role mapping for {organization_data.get('organization_name')}")
    items: List[Any] = [Part.from_bytes(data=b, mime_type="application/pdf") for b in (additional_document_contents or [])]
    items.append(_build_role_mapping_prompt_v1(organization_data))

    response = await get_llm().generate(
        [Message.user(*items)], model=LEGACY_PRO_MODEL, config=GenerationConfig(temperature=0.5)
    )
    if not response.text:
        raise Exception("Empty response from LLM")
    return json.loads(_strip_code_fences(response.text))


@traced_task()
async def stream_role_mapping_v1(
    organization_data: Dict[str, Any],
    additional_document: bytes | None = None,
) -> AsyncIterator[dict]:
    """Streaming variant of generate_role_mapping_v1. Yields {"type": "chunk"|"final", ...}."""
    items: List[Any] = []
    if additional_document:
        items.append(Part.from_bytes(data=additional_document, mime_type="application/pdf"))
    items.append(_build_role_mapping_prompt_v1(organization_data))

    buffer: List[str] = []
    async for chunk in get_llm().stream(
        [Message.user(*items)], model=LEGACY_PRO_MODEL, config=GenerationConfig(temperature=0.5)
    ):
        buffer.append(chunk)
        yield {"type": "chunk", "data": chunk}
    yield {"type": "final", "data": json.loads(_strip_code_fences("".join(buffer)))}


@traced_task()
async def generate_designation_role_mapping_v1(
    org_name: str, dep_name: str, designation: str, sector: str, instruction: str,
    acbp_summary: str, work_allocation_summary: str,
) -> Any:
    """v1 single-designation role mapping (the add-designation endpoint)."""
    output_format = {
        "designation_name": "[Designation Name]", "wing_division_section": "[Wing/Division/Section]",
        "role_responsibilities": "[List of Role Responsibilities]", "activities": "[List of Activities]",
        "competencies": [{"type": "[Behavioral/Functional/Domain]", "theme": "[Competency Theme]", "sub_theme": "[Competency Sub-theme]"}],
        "source": "[ACBP, Work Allocation Order, KCM, AI Suggested]",
    }
    prompt = DESIGNATION_ROLE_MAPPING_PROMPT_V1.format(
        organization_name=org_name, department_name=dep_name, designation_name=designation,
        sector=sector, instructions=instruction,
        acbp_summary=acbp_summary, work_allocation_summary=work_allocation_summary,
        kcm_competencies=json.dumps(_KCM_COMPETENCIES, indent=2),
        output_json_format=json.dumps(output_format, indent=None, separators=(",", ":")),
    )
    return await get_llm().generate_structured(
        [Message.user(prompt)], model=LEGACY_PRO_MODEL,
        schema=_DESIGNATION_ROLE_MAPPING_SCHEMA_V1, config=GenerationConfig(temperature=0.5),
    )


# ---------------------------------------------------------------------------
# Role mapping — v2
# ---------------------------------------------------------------------------

@traced_task()
async def generate_role_mapping_v2(organization_data: Dict[str, Any]) -> Any:
    """v2 single-pass role mapping. Returns the parsed JSON (a list of mappings)."""
    is_state = _is_state(organization_data)
    logger.info(f"Generating role mapping for {organization_data.get('organization_name')}")
    logger.info(f"Role Mapping is using prompt :: {'STATE_PROMPT' if is_state else 'CENTER_PROMPT'}")
    prompt_template = ROLE_MAPPING_PROMPT_V2_STATE if is_state else ROLE_MAPPING_PROMPT_V2_CENTRE
    prompt = prompt_template.format(
        organization_name=organization_data.get("organization_name"),
        department_name=organization_data.get("department_name"),
        instructions=organization_data.get("instruction"),
        primary_summary=organization_data.get("docs_summary"),
        kcm_competencies=json.dumps(_KCM_COMPETENCIES, indent=2),
        output_json_format=json.dumps(_V2_OUTPUT, indent=2),
    )
    response = await get_llm().generate(
        [Message.user(prompt)], model=LEGACY_PRO_MODEL, config=GenerationConfig(temperature=0.5)
    )
    if not response.text:
        raise Exception("Empty response from LLM")
    return json.loads(_strip_code_fences(response.text))


@traced_task()
async def generate_designation_role_mapping_v2(
    org_name: str, dep_name: str, designation: str, sector: str, instruction: str,
    primary_summary: str,
) -> Any:
    """v2 single-designation role mapping (the add-designation endpoint)."""
    logger.info(f"Generating role and competencies for designation: {designation}")
    output_format = {
        "designation_name": "[Designation Name]", "wing_division_section": "[Wing/Division/Section]",
        "role_responsibilities": "[List of Role Responsibilities]", "activities": "[List of Activities]",
        "competencies": [{"type": "[Behavioural/Functional/Domain]", "theme": "[Competency Theme]", "sub_theme": "[Competency Sub-theme]"}],
        "source": "[Primary document summaries, KCM, AI Suggested]",
    }
    prompt = DESIGNATION_ROLE_MAPPING_PROMPT_V2.format(
        organization_name=org_name, department_name=dep_name, designation_name=designation,
        sector=sector, instructions=instruction, primary_summary=primary_summary or "N/A",
        kcm_competencies=json.dumps(_KCM_COMPETENCIES, indent=2),
        output_json_format=json.dumps(output_format, indent=None, separators=(",", ":")),
    )
    return await get_llm().generate_structured(
        [Message.user(prompt)], model=settings.GEMINI_PRO_MODEL_NAME,
        schema=_DESIGNATION_ROLE_MAPPING_SCHEMA_V2, config=GenerationConfig(temperature=0.5),
    )


# ---------------------------------------------------------------------------
# Role mapping — v3 (multi-pass)
# ---------------------------------------------------------------------------

@traced_task()
async def extract_designations(organization_data: Dict[str, Any]) -> DesignationExtractionResponse:
    """v3 PASS 1: extract every unique designation, hierarchically ordered, from the WAO
    document summaries. Uses the FLASH model — factual extraction, no creativity."""
    logger.info(f"PASS 1: Extracting designations for {organization_data.get('organization_name')}")
    prompt = """
                Here is the input Data:
                Ministry/State Name: {ORGANIZATION_NAME}
                Department/Organisation Name: {DEPARTMENT_NAME}

                Primary reference document Summaries:
                {DOCUMENT_SUMMARIES}

                Extract ALL unique designations from the provided input data and organize them hierarchically based on the system prompt.
                """.format(
        ORGANIZATION_NAME=organization_data.get("organization_name"),
        DEPARTMENT_NAME=organization_data.get("department_name"),
        DOCUMENT_SUMMARIES=organization_data.get("docs_summary", "N/A"),
    )
    config = GenerationConfig(
        system_instruction=DESIGNATION_EXTRACTION_PROMPT,
        temperature=0.1,  # Very low — factual extraction, no creativity
        top_p=0.85,       # Restrict to high-probability tokens
    )
    return await get_llm().generate_structured(
        [Message.user(prompt)], model=settings.GEMINI_FLASH_MODEL_NAME,
        schema=DesignationExtractionResponse, config=config,
    )


@traced_task()
async def generate_frac_batch(
    designations_batch: List[Dict[str, Any]],
    organization_data: Dict[str, Any],
    batch_number: int,
) -> FRACBatchResponse:
    """v3 PASS 2: generate FRAC mappings for one batch of designations."""
    is_state = _is_state(organization_data)
    logger.info(f"PASS 2 - Batch {batch_number}: Processing {len(designations_batch)} designations")
    logger.info(f"Role Mapping is using prompt :: {'STATE_PROMPT' if is_state else 'CENTER_PROMPT'}")
    prompt_template = ROLE_MAPPING_PROMPT_V3_STATE if is_state else ROLE_MAPPING_PROMPT_V3_CENTRE
    output_format = _V3_STATE_OUTPUT if is_state else _V3_CENTRE_OUTPUT

    designation_context = json.dumps({
        "validated_designations": designations_batch,
        "batch_info": {"batch_number": batch_number, "total_in_batch": len(designations_batch)},
    }, indent=2)

    prompt = prompt_template.format(
        pass1_output=designation_context,
        organization_name=organization_data.get("organization_name"),
        department_name=organization_data.get("department_name"),
        instructions=organization_data.get("instruction"),
        primary_summary=organization_data.get("docs_summary"),
        kcm_competencies=json.dumps(_KCM_COMPETENCIES_V3, indent=2),
        output_json_format=json.dumps(output_format, indent=2),
    )
    return await get_llm().generate_structured(
        [Message.user(prompt)], model=settings.GEMINI_PRO_MODEL_NAME,
        schema=FRACBatchResponse, config=GenerationConfig(temperature=0.3, top_p=0.90),
    )


# ---------------------------------------------------------------------------
# Course recommendation
# ---------------------------------------------------------------------------

@traced_task()
async def embed_search_query(text: str) -> List[float]:
    """Embed one search query for course vector search. Returns [] for blank input or on
    failure, matching the caller's existing tolerance for missing embeddings."""
    if not text.strip():
        logger.warning("Attempted to get embedding for empty text; returning empty list")
        return []
    try:
        vectors = await get_embedder().embed_batch(
            [f"task: search result | query: {text}"],
            model=settings.GOOGLE_EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_OUTPUT_DIMENSIONALITY,
        )
        return vectors[0] if vectors else []
    except Exception as e:
        logger.exception(f"Error generating embedding for text '{text[:50]}...': {e}")
        return []


@traced_task()
async def generate_contextual_queries(user_profile: str) -> Dict[str, Any]:
    """Derive three vector-search queries plus a keyword list from a role profile:
    keyword_query, description_query, combined_query, search_keywords."""
    logger.info("Generating contextual queries from user profile")
    config = GenerationConfig(
        temperature=0.4, top_p=0.95, safety=SafetyPolicy.PERMISSIVE,
        system_instruction=VECTOR_QUERY_SYSTEM_PROMPT,
    )
    result = await get_llm().generate_structured(
        [Message.user(f"Role Profile:\n{user_profile}")], model=settings.GEMINI_PRO_MODEL_NAME,
        schema=_CONTEXTUAL_QUERIES_SCHEMA, config=config,
    )
    logger.info("Contextual queries generated successfully")
    return result


@traced_task()
async def infer_designation_group(user_profile: str) -> str:
    """Classify a role profile into Group A/B (senior/gazetted) or C/D (supporting/clerical).
    Defaults to "AB" on any failure, since this only biases retrieval."""
    config = GenerationConfig(
        temperature=0, max_output_tokens=256, safety=SafetyPolicy.PERMISSIVE,
        system_instruction=DESIGNNATION_GROUP_SYSTEM_PROMPT,
    )
    try:
        result = await get_llm().generate_structured(
            [Message.user(f"Role Profile:\n{user_profile}")], model=settings.GEMINI_FLASH_MODEL_NAME,
            schema=_DESIGNATION_GROUP_SCHEMA, config=config,
        )
        group = result.get("group", "AB")
        logger.info(f"LLM classified designation group as: {group}")
        return group
    except Exception as e:
        logger.warning(f"Designation group inference failed, defaulting to AB: {e}")
        return "AB"


@traced_task()
async def filter_courses(
    courses_prompt: str,
    user_profile: str,
    organisation: str,
    designation_group: str | None = None,
) -> str:
    """
    LLM-based course selection and scoring with:
    - Provider priority (own-org courses preferred)
    - Domain-mix enforcement by designation group
    - Sector-specific domain inclusion
    - Topic/type diversity within domain courses
    """
    logger.info("Filtering candidate courses through LLM")

    if designation_group == "AB":
        mix_rule = "Domain: ≥50%, Behavioral: ~25%, Functional: ~25%"
    else:
        mix_rule = "Domain: ~40%, Behavioral: ~30%, Functional: ~30%"
    # ({mix_rule})

    contents = [Message.user(f"""
Role Profile:
{user_profile}

Own Organisation: {organisation or 'N/A'}

Candidate Courses:
{courses_prompt}
""")]
    config = GenerationConfig(
        temperature=0, top_p=1, safety=SafetyPolicy.PERMISSIVE,
        json_output=True, response_schema=_COURSE_FILTER_SCHEMA,
        system_instruction=COURSE_SELECTION_SYSTEM_PROMPT,
        thinking_budget=2048, include_thoughts=False,
    )
    response = await get_llm().generate(contents, model=settings.GEMINI_PRO_MODEL_NAME, config=config)
    if not response.text:
        logger.error(f"LLM filtering empty response — failed to inspect: {response.raw}")
        return "[]"
    return response.text


@traced_task()
async def fetch_general_courses(user_profile: str) -> List[Dict[str, Any]]:
    """Web-search public courses across external learning platforms.

    OFF by default (ENABLE_GENERAL_COURSE_LOOKUP), preserving the behaviour of the
    unconditional `return []` this replaced — the lookup was deliberately disabled, so leaving
    it enabled would silently add a web-search call per recommendation and mix public
    (non-iGOT) courses into results.

    Also requires the provider to support a builtin web-search tool; returns [] when it does
    not, rather than issuing a search-less request that would invent course links.
    """
    if not settings.ENABLE_GENERAL_COURSE_LOOKUP:
        return []

    llm = get_llm()
    if not llm.supports(Capability.WEB_SEARCH):
        logger.warning(
            f"Provider '{llm.name}' has no builtin web search; skipping general course lookup "
            "(a search-less request would fabricate course URLs)."
        )
        return []

    logger.info("Fetching the general courses across the learning platforms")
    config = GenerationConfig(
        system_instruction=_GENERAL_COURSES_SYSTEM_PROMPT, temperature=0.5,
        tools=[Tool.WEB_SEARCH], safety=SafetyPolicy.HATE_SPEECH_ONLY,
    )
    try:
        response = await llm.generate(
            [Message.user(f"Here's the user role context: {user_profile}")],
            model=settings.GEMINI_PRO_MODEL_NAME, config=config,
        )
        if not response.text:
            logger.warning("General courses response was empty")
            return []
        general_courses = json.loads(_strip_code_fences(response.text))
        for course in general_courses:
            course["identifier"] = str(uuid.uuid4())
            course["is_public"] = True
        logger.info(f"Fetched {len(general_courses)} general courses")
        return general_courses
    except Exception as e:
        logger.exception(f"Error fetching general courses: {e}")
        return []


# ---------------------------------------------------------------------------
# Designation matching
# ---------------------------------------------------------------------------

# Vector width for designation matching. Fixed at 768 because that is the width of the
# designation_embeddings.embedding pgvector column AND the width
# scripts/ingest_designation_embeddings.py writes — query and stored vectors must match
# exactly. Deliberately NOT settings.EMBEDDING_OUTPUT_DIMENSIONALITY (1536), which belongs to
# the unrelated course-search vector space. Changing this requires re-ingesting the table.
DESIGNATION_EMBEDDING_DIMENSIONS = 768

# The prefix must stay byte-identical to format_for_matching() in
# scripts/ingest_designation_embeddings.py, or queries land in a different region of the
# embedding space than the stored vectors and similarity scores quietly degrade.
_DESIGNATION_EMBED_PREFIX = "task: sentence similarity | query: "


@traced_task()
async def embed_designations(designations: List[str]) -> List[List[float]]:
    """Embed designation names for pgvector similarity matching against the iGOT designation
    master. Redis caching lives in the caller, which owns the cache keying."""
    return await get_embedder().embed(
        [f"{_DESIGNATION_EMBED_PREFIX}{d}" for d in designations],
        model=settings.GOOGLE_EMBEDDING_MODEL,
        dimensions=DESIGNATION_EMBEDDING_DIMENSIONS,
        batch_size=settings.DESIGNATION_EMBED_BATCH_SIZE,
    )

