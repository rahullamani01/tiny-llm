"""Checks for held-out coverage and fixed-prompt evaluation."""
import torch

from chat.template import CHAT_SPECIAL_TOKENS, END
from data.filter_chat_holdout import filter_known
from experiments.compare_chat import complete
from tokenizer import CharTokenizer


def test_holdout_filter_uses_only_training_characters():
    def conv(text):
        return {"system": None, "messages": [{"role": "user", "content": text}]}
    kept, counts = filter_known([conv("ab")], [conv("aba"), conv("abé")])
    assert kept == [conv("aba")]
    assert counts == {"train": 1, "validation": 2, "retained": 1, "excluded": 1}


def test_compare_stops_on_special_token():
    tok = CharTokenizer("abc", special_tokens=CHAT_SPECIAL_TOKENS)

    class FixedModel(torch.nn.Module):
        block_size = 32

        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))

        def generate_stream(self, idx, max_new_tokens, temperature):
            for token in [tok.encode("a")[0], tok.special_id(END)]:
                idx = torch.cat((idx, torch.tensor([[token]], device=idx.device)), dim=1)
                yield idx

    model = FixedModel()
    result = complete(model, tok, "abc", 5, chat=True, system=None)
    assert result == {"text": "a", "stop": "special_token"}
    assert not model.training
