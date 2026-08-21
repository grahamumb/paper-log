"""The vision-model layer: one small seam between paper-log and a VLM.

Everything that knows the name of a vendor lives here. The rest of paper-log
asks for "the text on this image" and gets a :class:`Reply`, which is why
swapping or adding a provider is a class in this file rather than a rewrite
somewhere else.

Two things earn the abstraction rather than just being indirection:

*Image limits are per model, not per product.* A vision model resizes anything
past its limit server-side, which silently throws away the resolution you were
counting on -- and on handwriting resolution is most of the game. The limits
differ sharply between tiers (a 2576px model sees more than twice the pixel area
of a 1568px one), so the sizing has to be looked up from the model, not
hard-coded once. See :data:`MODEL_LIMITS`.

*Configuration has to reach the endpoint.* Pointing at a gateway, a proxy, a
self-hosted endpoint or a different vendor is a base URL and a key, and none of
that should require touching transcription logic.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: Long edge in pixels, and total pixels, that a model accepts before it starts
#: resizing on its own. Keyed by a prefix of the model id, longest match first.
#:
#: These are a convenience, not an authority -- the Models API is the authority,
#: and a model released after this table was written will fall through to
#: :data:`CONSERVATIVE_LIMITS`. Both numbers are overridable in config, which is
#: the escape hatch when this table is wrong or stale.
MODEL_LIMITS: Dict[str, Tuple[int, int]] = {
    # High-resolution tier.
    "claude-opus-5": (2576, 3_750_000),
    "claude-opus-4-8": (2576, 3_750_000),
    "claude-opus-4-7": (2576, 3_750_000),
    "claude-sonnet-5": (2576, 3_750_000),
    "claude-fable-5": (2576, 3_750_000),
    "claude-mythos-5": (2576, 3_750_000),
    # Earlier tier. Haiku 4.5 is here, which is the case that matters in
    # practice: it is the obvious model to reach for on cost, and it sees
    # roughly 45% of the pixel area the high-resolution models do.
    "claude-haiku-4-5": (1568, 1_150_000),
    "claude-opus-4-6": (1568, 1_150_000),
    "claude-opus-4-5": (1568, 1_150_000),
    "claude-sonnet-4-6": (1568, 1_150_000),
    "claude-sonnet-4-5": (1568, 1_150_000),
}

#: Used for a model this file has never heard of. Deliberately the smaller
#: tier: sending less than a model could take costs a little accuracy, whereas
#: sending more than it takes means it resizes to a size nobody chose. Of the
#: two, the guessable failure is better than the invisible one.
CONSERVATIVE_LIMITS: Tuple[int, int] = (1568, 1_150_000)

DEFAULT_MODEL = "claude-opus-5"

#: Transcription is reading, not reasoning. Low effort keeps the thinking spend
#: proportionate -- and is much preferred to turning thinking off, which on
#: Opus 5 can leak internal tags into the response.
DEFAULT_EFFORT = "low"
EFFORTS = ("low", "medium", "high", "xhigh", "max")

#: Thinking shares this budget with the reply on current models, so it has to
#: cover both. A dense handwritten page is well under 2000 tokens of text.
DEFAULT_MAX_TOKENS = 8000

PROVIDERS = ("anthropic",)


class VisionError(RuntimeError):
    """Raised when a vision model cannot be reached or will not answer."""


@dataclass(frozen=True)
class Reply:
    """What a vision model said, with the vendor's shapes already peeled off."""

    text: str
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


@dataclass
class VisionSpec:
    """Which model to ask, how, and where.

    Every field is settable from config, so switching model or endpoint never
    means editing code.
    """

    provider: str = "anthropic"
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    #: Endpoint override -- a gateway, a proxy, a local stub. None means the
    #: SDK's own default, which also honours ANTHROPIC_BASE_URL.
    base_url: Optional[str] = None
    #: Which environment variable holds the key. Named rather than valued so a
    #: config file can be committed without becoming a secret.
    api_key_env: str = "ANTHROPIC_API_KEY"
    #: Override the model's image limits. None means look them up.
    max_edge: Optional[int] = None
    max_pixels: Optional[int] = None
    timeout: float = 300.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise VisionError(
                f"unknown provider {self.provider!r}; built in: {', '.join(PROVIDERS)}"
            )
        if self.effort not in EFFORTS:
            raise VisionError(
                f"effort must be one of {', '.join(EFFORTS)}, got {self.effort!r}"
            )
        for name in ("max_tokens", "timeout"):
            if getattr(self, name) <= 0:
                raise VisionError(f"{name} must be greater than zero")
        for name in ("max_edge", "max_pixels"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise VisionError(f"{name} must be greater than zero when set")

    @property
    def limits(self) -> Tuple[int, int]:
        """The image limits to size for: config wins, else the model's own."""
        edge, pixels = limits_for(self.model)
        return (self.max_edge or edge, self.max_pixels or pixels)

    @property
    def limits_are_a_guess(self) -> bool:
        """True when we fell back rather than recognising the model.

        Worth surfacing: it means the image may be smaller than the model would
        have accepted, which on handwriting costs accuracy.
        """
        return (
            self.max_edge is None
            and self.max_pixels is None
            and _match_limits(self.model) is None
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionSpec":
        known = {item.name for item in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise VisionError(
                f"unknown vision setting(s) {', '.join(sorted(unknown))}; "
                f"known: {', '.join(sorted(known))}"
            )
        return cls(**data)


def _normalise(model: str) -> str:
    """Strip the decorations that identify a deployment rather than a model.

    Bedrock prefixes with the vendor, Vertex separates a dated snapshot with an
    ``@``, and gateways sometimes prefix a route. The tier is the same
    underneath, so the limits are too.
    """
    text = model.strip().lower()
    text = text.split("@", 1)[0]
    for prefix in ("anthropic.", "anthropic/", "us.anthropic.", "eu.anthropic."):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text


def _match_limits(model: str) -> Optional[Tuple[int, int]]:
    text = _normalise(model)
    # Longest prefix first, so claude-opus-4-8 is not caught by claude-opus-4.
    for key in sorted(MODEL_LIMITS, key=len, reverse=True):
        if text.startswith(key):
            return MODEL_LIMITS[key]
    return None


def limits_for(model: str) -> Tuple[int, int]:
    """(long edge, total pixels) a model accepts before it resizes for you."""
    return _match_limits(model) or CONSERVATIVE_LIMITS


def encode_array(array, *, max_edge: int, max_pixels: int) -> Tuple[str, str, Tuple[int, int]]:
    """Encode an in-memory image the same way a file on disk would be.

    Used for a region cut out of a page: the crop never touches the filesystem,
    which is one fewer place for the pixels around it to come back.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise VisionError(
            "reading images needs Pillow: pip install 'paper-log[transcribe]'"
        ) from exc
    import numpy

    data = numpy.asarray(array)
    if data.ndim == 3:
        image = Image.fromarray(data[:, :, ::-1], "RGB")
    else:
        image = Image.fromarray(data.astype("uint8"), "L")
    return _encode(image, max_edge=max_edge, max_pixels=max_pixels)


def encode_image(
    path: Path, *, max_edge: int, max_pixels: int
) -> Tuple[str, str, Tuple[int, int]]:
    """Read an image, fit it to the limits, return (media_type, base64, size).

    PNG rather than JPEG: the strokes are thin and high-contrast, which is
    exactly where JPEG puts its ringing, and a flattened page compresses well
    losslessly because most of it is flat white.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise VisionError(
            "reading images needs Pillow: pip install 'paper-log[transcribe]'"
        ) from exc

    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise VisionError(f"cannot read {Path(path).name}: {exc}") from exc

    return _encode(image, max_edge=max_edge, max_pixels=max_pixels)


def _encode(image, *, max_edge: int, max_pixels: int) -> Tuple[str, str, Tuple[int, int]]:
    from PIL import Image

    if image.mode not in ("L", "RGB"):
        image = image.convert("L")

    # Whichever limit binds harder wins; both are satisfied by one resize.
    scale = min(
        max_edge / max(image.size),
        (max_pixels / (image.width * image.height)) ** 0.5,
        1.0,
    )
    if scale < 1.0:
        # Round down, not to nearest: rounding up on both axes can put the
        # result back over the area limit by a few hundred pixels, which is
        # exactly the silent server-side resize this is here to avoid.
        image = image.resize(
            (max(int(image.width * scale), 1), max(int(image.height * scale), 1)),
            Image.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return "image/png", data, image.size


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


class AnthropicBackend:
    """Reads an image with the Claude Messages API."""

    def __init__(self, spec: VisionSpec, client=None):
        self.spec = spec
        self._client = client

    @property
    def limits(self) -> Tuple[int, int]:
        """Image limits for the model this backend will actually call.

        The backend is the single authority for anything model-derived, because
        it is the thing holding the model name that goes on the wire. Reading
        limits off a config instead invites the two to disagree -- and a
        disagreement here is invisible: the image is simply the wrong size.
        """
        return self.spec.limits

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def preflight(self) -> None:
        """Build the client now, so a bad setup fails once instead of per page.

        Without this, a missing key surfaces as an identical failure on every
        page in the run -- fifty lines of the same error, and a caller that has
        to read to the end to find out nothing worked.
        """
        _ = self.client

    def _build_client(self):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise VisionError(
                "transcription needs the anthropic package: "
                "pip install 'paper-log[transcribe]'"
            ) from exc

        key = os.environ.get(self.spec.api_key_env)
        if not key and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            raise VisionError(
                f"no API key: set {self.spec.api_key_env} in your environment. "
                "There is deliberately no --api-key flag, because a key on the "
                "command line ends up in your shell history and in `ps`. Keys "
                "are at https://console.anthropic.com/settings/keys"
            )
        options: Dict[str, Any] = {
            "timeout": self.spec.timeout,
            "max_retries": self.spec.max_retries,
        }
        if key:
            options["api_key"] = key
        if self.spec.base_url:
            options["base_url"] = self.spec.base_url
        return anthropic.Anthropic(**options)

    def read(self, *, media_type: str, data: str, system: str, instruction: str) -> Reply:
        """Ask about an image."""
        return self._send(
            system,
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                },
                {"type": "text", "text": instruction},
            ],
        )

    def ask(self, *, system: str, instruction: str) -> Reply:
        """Ask in text alone, with no image in the request at all.

        Used to run a tool on a prompt lifted off a page. The page is
        deliberately absent: the request travels as the writer's words, and
        nothing that was next to them on the paper travels with it.
        """
        return self._send(system, [{"type": "text", "text": instruction}])

    def _send(self, system: str, content) -> Reply:
        message = self.client.messages.create(
            model=self.spec.model,
            max_tokens=self.spec.max_tokens,
            system=system,
            output_config={"effort": self.spec.effort},
            messages=[{"role": "user", "content": content}],
        )

        # Read the stop reason before touching content: a refusal comes back as
        # a perfectly ordinary 200 with nothing in it, and indexing blindly
        # would turn that into an IndexError far from anything meaningful.
        stop_reason = getattr(message, "stop_reason", None) or "end_turn"
        blocks = getattr(message, "content", None) or []
        text = "\n".join(
            block.text
            for block in blocks
            if getattr(block, "type", None) == "text" and getattr(block, "text", "")
        ).strip()
        usage = getattr(message, "usage", None)
        return Reply(
            text=text,
            stop_reason=stop_reason,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=getattr(message, "model", "") or self.spec.model,
        )


BACKENDS = {"anthropic": AnthropicBackend}


def backend_for(spec: VisionSpec, client=None):
    """Build the backend a spec asks for."""
    try:
        factory = BACKENDS[spec.provider]
    except KeyError:
        raise VisionError(
            f"no backend for provider {spec.provider!r}; built in: "
            f"{', '.join(sorted(BACKENDS))}"
        ) from None
    return factory(spec, client=client)
