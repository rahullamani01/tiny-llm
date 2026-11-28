"""UTF-8 tokenizer, streaming and checkpoint compatibility."""
import os
import tempfile

import torch

from chat.template import CHAT_SPECIAL_TOKENS, END
from checkpoint import save_checkpoint
from inference.generate import load_model, stream_text
from model.model import TinyGPT
from server.chat import iter_chat_events
from tokenizer import ByteTokenizer, incremental_decoder, tokenizer_from_state, with_special_tokens


def test_byte_unicode_roundtrip_and_state():
    tok = ByteTokenizer()
    text = "Hello ´ × 漢字 😀"
    assert tok.decode(tok.encode(text)) == text
    assert tok.missing_chars(text) == []
    assert tokenizer_from_state(tok.to_dict()).encode(text) == tok.encode(text)
    assert tok.vocab_size == 256


def test_byte_incremental_utf8_and_invalid_bytes():
    tok = ByteTokenizer(CHAT_SPECIAL_TOKENS)
    decoder = incremental_decoder(tok)
    encoded = tok.encode("😀")
    assert [decoder.push(i) for i in encoded] == ["", "", "", "😀"]
    assert decoder.finish() == ""
    assert tok.decode([255]) == "\ufffd"
    assert tok.decode([0xC3, tok.special_id(END)]) == "\ufffd" + END
    added = with_special_tokens(ByteTokenizer(), CHAT_SPECIAL_TOKENS)
    assert added.encode("😀") == encoded
    assert added.special_id(END) >= 256


def test_byte_stream_and_checkpoint():
    tok = ByteTokenizer()
    cfg = dict(vocab_size=256, block_size=8, n_embd=8, n_head=2, n_layer=1)
    model = TinyGPT(**cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "byte.pt")
        save_checkpoint(path, model, None, cfg, tok.chars, 0, [], 1.0,
                        tokenizer=tok.to_dict())
        loaded, loaded_tok = load_model(path, torch.device("cpu"))
        assert loaded_tok.decode(loaded_tok.encode("😀")) == "😀"
        assert loaded.block_size == 8

    class Fixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def generate_stream(self, idx, *args, **kwargs):
            for token in tok.encode("😀"):
                idx = torch.cat((idx, torch.tensor([[token]])), dim=1)
                yield idx

    assert "".join(stream_text(Fixed(), tok, "x", max_new_tokens=4)) == "😀"


def test_byte_server_stream_preserves_unicode():
    tok = ByteTokenizer(CHAT_SPECIAL_TOKENS)

    class Fixed(torch.nn.Module):
        block_size = 32

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def generate_stream(self, idx, *args, **kwargs):
            for token in tok.encode("é😀") + [tok.special_id(END)]:
                idx = torch.cat((idx, torch.tensor([[token]], device=idx.device)), dim=1)
                yield idx

    events = list(iter_chat_events(Fixed(), tok, [{"role": "user", "content": "hi"}]))
    assert "".join(e["token"] for e in events[:-1]) == "é😀"
    assert events[-1] == {"done": True, "content": "é😀"}
