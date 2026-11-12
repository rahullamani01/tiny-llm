def _normalize_special_tokens(special_tokens):
    """Validate and de-duplicate special tokens, preserving order."""
    out = []
    for token in special_tokens or ():
        if not isinstance(token, str) or not token:
            raise ValueError("special tokens must be non-empty strings")
        if token not in out:
            out.append(token)
    return out


class CharTokenizer:
    def __init__(self, text: str, special_tokens=()):
        self.chars = sorted(set(text))                         # every unique character
        self.special_tokens = _normalize_special_tokens(special_tokens)
        overlap = set(self.chars) & set(self.special_tokens)
        if overlap:
            shown = ", ".join(repr(t) for t in sorted(overlap))
            raise ValueError(f"special tokens overlap with characters: {shown}")

        self.stoi = {c: i for i, c in enumerate(self.chars)}   # char -> id
        for token in self.special_tokens:                      # append after characters
            self.stoi[token] = len(self.stoi)
        self.itos = {i: c for c, i in self.stoi.items()}       # id -> char/token
        self.vocab_size = len(self.itos)

    @classmethod
    def from_chars(cls, chars, special_tokens=()):
        """Rebuild a tokenizer from a saved character list (used when loading checkpoints)."""
        return cls("".join(chars), special_tokens=special_tokens)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def special_id(self, token: str) -> int:
        if token not in self.special_tokens:
            raise KeyError(f"unknown special token {token!r}")
        return self.stoi[token]

    def is_special_id(self, token_id: int) -> bool:
        return token_id >= len(self.chars)

    def missing_chars(self, s: str) -> list[str]:
        """Characters in `s` that are not part of the vocabulary."""
        return sorted(set(s) - set(self.stoi))

    def to_dict(self):
        return {
            "type": "char",
            "chars": list(self.chars),
            "special_tokens": list(self.special_tokens),
        }

    @classmethod
    def from_dict(cls, state):
        return cls.from_chars(
            state["chars"], state.get("special_tokens", ())
        )
