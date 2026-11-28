"""A fixed 256-token UTF-8 byte vocabulary (not byte-level BPE)."""
import codecs

from tokenizer.tokenizer import _normalize_special_tokens


class ByteTokenizer:
    def __init__(self, special_tokens=()):
        self.special_tokens = _normalize_special_tokens(special_tokens)
        self.chars = list(range(256))  # compatibility with checkpoint metadata
        self.vocab_size = 256 + len(self.special_tokens)

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        decoder = self.incremental_decoder()
        return "".join(decoder.push(i) for i in ids) + decoder.finish()

    def special_id(self, token):
        if token not in self.special_tokens:
            raise KeyError(f"unknown special token {token!r}")
        return 256 + self.special_tokens.index(token)

    def is_special_id(self, token_id):
        return 256 <= token_id < self.vocab_size

    def missing_chars(self, text):
        # Python strings can contain unpaired surrogates, which are not UTF-8.
        text.encode("utf-8")
        return []

    def to_dict(self):
        return {"type": "byte", "special_tokens": list(self.special_tokens)}

    @classmethod
    def from_dict(cls, state):
        return cls(state.get("special_tokens", ()))

    def incremental_decoder(self):
        return _ByteDecoder(self)


class _ByteDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id):
        if 0 <= token_id < 256:
            return self.decoder.decode(bytes([token_id]), final=False)
        if self.tokenizer.is_special_id(token_id):
            prefix = self.decoder.decode(b"", final=True)
            self.decoder.reset()
            return prefix + self.tokenizer.special_tokens[token_id - 256]
        raise ValueError(f"unknown byte token id {token_id}")

    def finish(self):
        return self.decoder.decode(b"", final=True)
