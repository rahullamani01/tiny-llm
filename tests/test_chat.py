"""Tests for chat templates, special tokens, and the SFT data path."""
import os
import tempfile
import json
import subprocess
import sys

import torch

from chat.template import (
    ASSISTANT,
    CHAT_SPECIAL_TOKENS,
    END,
    SYSTEM,
    USER,
    build_chat_prompt,
    build_sft_example,
    normalize_messages,
)
from checkpoint import save_checkpoint
from model.model import TinyGPT
from tokenizer import (
    BPETokenizer,
    CharTokenizer,
    tokenizer_from_state,
    with_special_tokens,
)
from training.sft import (
    _resize_embeddings,
    encode_conversations,
    get_sft_batch,
    load_base_weights,
    load_conversations,
    split_conversations,
)


def _tok():
    text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?"
    return CharTokenizer(text, special_tokens=CHAT_SPECIAL_TOKENS)


def _conversation():
    return [
        {"role": SYSTEM, "content": "Be nice."},
        {"role": USER, "content": "hi"},
        {"role": ASSISTANT, "content": "hello"},
    ]


def test_special_token_roundtrip_char_and_bpe():
    tok = _tok()
    assert tok.vocab_size == len(tok.chars) + len(CHAT_SPECIAL_TOKENS)
    for token in CHAT_SPECIAL_TOKENS:
        token_id = tok.special_id(token)
        assert tok.is_special_id(token_id)
        assert tok.decode([token_id]) == token
    assert not tok.is_special_id(tok.encode("a")[0])

    again = tokenizer_from_state(tok.to_dict())
    assert again.vocab_size == tok.vocab_size
    assert again.special_tokens == list(CHAT_SPECIAL_TOKENS)

    base = BPETokenizer.train("abcabcabc", vocab_size=6, max_chars=9)
    bpe = with_special_tokens(base, CHAT_SPECIAL_TOKENS)
    assert bpe.vocab_size == base.vocab_size + len(CHAT_SPECIAL_TOKENS)
    assert bpe.encode("abc") == base.encode("abc")
    assert bpe.decode(bpe.encode("abc")) == "abc"
    assert bpe.special_id(END) >= base.vocab_size
    assert tokenizer_from_state(bpe.to_dict()).to_dict() == bpe.to_dict()


def test_template_masks_system_user_and_role_tokens():
    tok = _tok()
    ids, labels = build_sft_example(_conversation(), tok)
    assert len(ids) == len(labels)

    answer_ids = tok.encode("hello")
    answer_labels = [
        label for label in labels if label != -100
    ]
    assert answer_labels[: len(answer_ids)] == answer_ids
    assert answer_labels[-1] == tok.special_id(END)

    # System/user content and every role token are not trained.
    prompt_tokens = [
        tok.special_id("<|system|>"),
        *tok.encode("Be nice."),
        tok.special_id("<|user|>"),
        *tok.encode("hi"),
    ]
    positions = [ids.index(token) for token in prompt_tokens]
    assert all(labels[i] == -100 for i in positions)


def test_chat_prompt_includes_system_message():
    tok = _tok()
    ids = build_chat_prompt(
        [{"role": USER, "content": "hi"}], tok, block_size=48, system="system"
    )
    assert tok.special_id("<|system|>") in ids
    assert ids[-1] == tok.special_id("<|assistant|>")


def test_chat_prompt_fits_and_ends_with_assistant_role():
    tok = _tok()
    messages = [
        {"role": SYSTEM, "content": "system"},
        {"role": USER, "content": "a" * 20},
        {"role": ASSISTANT, "content": "b" * 20},
        {"role": USER, "content": "last"},
    ]
    ids = build_chat_prompt(messages, tok, block_size=24)
    assert len(ids) <= 24
    assert ids[-1] == tok.special_id("<|assistant|>")


def test_normalize_rejects_bad_messages():
    for messages in ([], [{"role": "bot", "content": "x"}], [{"role": USER, "content": ""}]):
        try:
            normalize_messages(messages)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"expected error for {messages!r}")


def test_sft_batch_has_supervision_and_finite_loss():
    tok = _tok()
    conversations = [
        {"messages": _conversation(), "system": None},
        {"messages": [{"role": USER, "content": "a"}, {"role": ASSISTANT, "content": "hello"}], "system": None},
        {"messages": [{"role": USER, "content": "b"}, {"role": ASSISTANT, "content": "hello"}], "system": None},
    ]
    train, val = split_conversations(conversations, val_fraction=0.34, seed=0)
    assert train and val
    ids, labels = encode_conversations(train, tok)
    x, y = get_sft_batch(ids, labels, block_size=8, batch_size=4, device="cpu")
    assert x.shape == y.shape == (4, 8)
    assert torch.all((y != -100).any(dim=1))

    model = TinyGPT(
        vocab_size=tok.vocab_size,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    _, loss = model(x, y)
    assert torch.isfinite(loss)


def test_sft_batch_predicts_next_token_even_in_fallback():
    ids = torch.arange(12)
    labels = torch.full_like(ids, -100)
    labels[11] = 11
    x, y = get_sft_batch(ids, labels, block_size=4, batch_size=32, device="cpu")
    assert (y != -100).any(dim=1).all()
    assert torch.all(y[y != -100] == 11)
    assert torch.all(x[:, -1] == 10)


def test_sft_split_keeps_matching_prompts_together():
    def conv(question, answer):
        return {"messages": [{"role": USER, "content": question},
                             {"role": ASSISTANT, "content": answer}]}
    conversations = [conv(" Hello  world ", "one"), conv("hello world", "two"),
                     conv("different", "three"), conv("other", "four")]
    train, val = split_conversations(conversations, seed=4)
    assert (conversations[0] in train) == (conversations[1] in train)
    assert (train, val) == split_conversations(conversations, seed=4)
    try:
        split_conversations(conversations[:2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected failure for a single prompt group")


def test_sft_cli_trains_with_separate_holdout():
    with tempfile.TemporaryDirectory() as tmp:
        train = os.path.join(tmp, "train.jsonl")
        val = os.path.join(tmp, "val.jsonl")
        for path, prompt in ((train, "abba"), (val, "baab")):
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "abababab"},
                ]}) + "\n")
        output = os.path.join(tmp, "chat.pt")
        subprocess.run([
            sys.executable, "-m", "training.sft", "--data", train, "--val-data", val,
            "--tokenizer", "char", "--out", output, "--block-size", "8",
            "--n-embd", "8", "--n-head", "2", "--n-layer", "1",
            "--batch-size", "2", "--max-steps", "2", "--warmup-steps", "0",
            "--eval-interval", "1", "--eval-iters", "1", "--device", "cpu", "--no-plot",
        ], check=True, capture_output=True)
        from checkpoint import load_checkpoint
        ckpt = load_checkpoint(output)
        assert ckpt["step"] == 2
        assert ckpt["train_config"]["sft_objective"] == "next_token_v1"


def test_load_conversations_and_split():
    lines = [
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}',
        '{"system":"s","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}',
        "",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "chat.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        convs = load_conversations(path)
        assert len(convs) == 2
        assert convs[0]["system"] is None
        assert convs[1]["system"] == "s"

        try:
            load_conversations(os.path.join(tmp, "missing.jsonl"))
        except OSError:
            pass
        else:
            raise AssertionError("expected OSError for a missing file")


def test_base_checkpoint_resizes_chat_vocabulary():
    base_tok = CharTokenizer("abc")
    base_cfg = dict(
        vocab_size=base_tok.vocab_size,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        tie_weights=False,
    )
    base = TinyGPT(**base_cfg)
    chat_tok = with_special_tokens(base_tok, CHAT_SPECIAL_TOKENS)
    chat_cfg = dict(base_cfg, vocab_size=chat_tok.vocab_size)
    model = TinyGPT(**chat_cfg)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "base.pt")
        save_checkpoint(
            path,
            base,
            None,
            base_cfg,
            base_tok.chars,
            0,
            [],
            float("inf"),
            tokenizer=base_tok.to_dict(),
        )
        _resize_embeddings(model, chat_tok.vocab_size)
        load_base_weights(model, path, "cpu")

    assert model.embed.token_embedding.weight.shape[0] == chat_tok.vocab_size
    torch.testing.assert_close(
        model.embed.token_embedding.weight[: base_tok.vocab_size],
        base.embed.token_embedding.weight,
    )
    torch.testing.assert_close(
        model.lm_head.weight[: base_tok.vocab_size],
        base.lm_head.weight,
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
