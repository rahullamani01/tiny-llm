"""Chat templates and SFT label construction.

The format is deliberately simple and readable:

    <|system|>You are a helpful assistant.<|end|>
    <|user|>Hi!<|end|>
    <|assistant|>Hello! How can I help?<|end|>

During SFT, every token is an input, but only assistant content and the final
``<|end|>`` token contribute to the loss. Everything else is labelled ``-100``,
which PyTorch's cross-entropy ignores.
"""
from dataclasses import dataclass

SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

ROLE_TOKENS = {
    SYSTEM: "<|system|>",
    USER: "<|user|>",
    ASSISTANT: "<|assistant|>",
}
END = "<|end|>"
PAD = "<|pad|>"
CHAT_SPECIAL_TOKENS = tuple(ROLE_TOKENS.values()) + (END, PAD)

IGNORE_INDEX = -100


@dataclass
class Message:
    role: str
    content: str


def normalize_messages(messages, system=None):
    """Validate a conversation and optionally prepend a system message.

    ``messages`` may be a list of ``{"role", "content"}`` dicts or ``Message``
    objects. The returned list always uses fresh ``Message`` objects.
    """
    if isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a sequence of message objects")

    out = []
    if system is not None:
        if not isinstance(system, str):
            raise TypeError("system must be a string")
        if not system.strip():
            raise ValueError("system prompt must not be empty")
        existing_system = False
        for message in messages:
            if isinstance(message, Message):
                existing_system = existing_system or message.role == SYSTEM
            elif isinstance(message, dict):
                existing_system = existing_system or message.get("role") == SYSTEM
        if not existing_system:
            out.append(Message(SYSTEM, system.strip()))

    for message in messages:
        if isinstance(message, Message):
            role, content = message.role, message.content
        elif isinstance(message, dict):
            role, content = message.get("role"), message.get("content")
        else:
            raise TypeError("each message must be a dict or Message")

        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role {role!r}; expected system/user/assistant")
        if not isinstance(content, str):
            raise TypeError(f"message content for role {role!r} must be a string")
        if not content.strip():
            raise ValueError(f"message content for role {role!r} must not be empty")
        out.append(Message(role, content.strip()))

    if not out:
        raise ValueError("a conversation needs at least one message")
    return out


def _role_id(tokenizer, role):
    return tokenizer.special_id(ROLE_TOKENS[role])


def render_ids(messages, tokenizer, add_generation_prompt=False):
    """Render normalized messages to token ids (special tokens included)."""
    ids = []
    for message in messages:
        ids.append(_role_id(tokenizer, message.role))
        ids.extend(tokenizer.encode(message.content))
        ids.append(tokenizer.special_id(END))
    if add_generation_prompt:
        ids.append(_role_id(tokenizer, ASSISTANT))
    return ids


def build_sft_example(messages, tokenizer, system=None):
    """Return ``(input_ids, labels)`` for one conversation.

    The two lists have the same length. System/user tokens and every role token
    are masked out of the loss. Assistant content and its closing ``<|end|>``
    token are trained normally.
    """
    messages = normalize_messages(messages, system=system)
    ids, labels = [], []

    for message in messages:
        content_ids = tokenizer.encode(message.content)
        role_id = _role_id(tokenizer, message.role)
        end_id = tokenizer.special_id(END)

        ids.extend([role_id, *content_ids, end_id])
        if message.role == ASSISTANT:
            labels.extend([IGNORE_INDEX, *content_ids, end_id])
        else:
            labels.extend([IGNORE_INDEX] * (len(content_ids) + 2))

    return ids, labels


def build_chat_prompt(
    messages, tokenizer, block_size, add_generation_prompt=True, system=None
):
    """Render a chat prompt that fits the model's context window.

    ``system`` is prepended when provided. Old non-system turns are dropped
    first. If a single remaining turn is still too long, the oldest tokens are
    truncated.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    messages = normalize_messages(messages, system=system)
    while True:
        ids = render_ids(messages, tokenizer, add_generation_prompt=add_generation_prompt)
        if len(ids) <= block_size:
            return ids

        drop = next(
            (i for i, message in enumerate(messages) if message.role != SYSTEM),
            None,
        )
        if drop is None:
            break
        del messages[drop]

    # Even the system prompt alone is too long: keep the newest tokens.
    return render_ids(
        messages, tokenizer, add_generation_prompt=add_generation_prompt
    )[-block_size:]
