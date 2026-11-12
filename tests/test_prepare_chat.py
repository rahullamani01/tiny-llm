"""Tests for instruction-data conversion used by the chat training recipe."""
import os
import tempfile

from data.prepare_chat import (
    clean_text,
    conversation_to_text,
    convert_records,
    load_records,
    prepare,
    record_to_conversation,
    write_chat_jsonl,
    write_text,
)
from training.sft import load_conversations


def _alpaca():
    return {
        "instruction": "Summarize the passage.",
        "input": "Cats sleep a lot.",
        "output": "Cats sleep often.",
    }


def _dolly():
    return {
        "instruction": "Where was Ada Lovelace born?",
        "context": "Ada Lovelace was born in London.",
        "response": "London.",
    }


def test_clean_text_normalizes_whitespace():
    assert clean_text(None) == ""
    assert clean_text("  a\r\nb  ") == "a\nb"
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_alpaca_record_becomes_user_assistant():
    conv = record_to_conversation(_alpaca())
    assert conv["messages"][0]["role"] == "user"
    assert conv["messages"][0]["content"] == (
        "Summarize the passage.\n\nInput:\nCats sleep a lot."
    )
    assert conv["messages"][1] == {"role": "assistant", "content": "Cats sleep often."}
    assert conv["system"]


def test_dolly_record_uses_context_label():
    conv = record_to_conversation(_dolly(), system="Be brief.")
    assert conv["system"] == "Be brief."
    assert "Context:\nAda Lovelace" in conv["messages"][0]["content"]
    assert conv["messages"][1]["content"] == "London."


def test_invalid_and_long_records_are_dropped():
    assert record_to_conversation(None) is None
    assert record_to_conversation({"instruction": "", "output": "x"}) is None
    assert record_to_conversation({"instruction": "q", "output": ""}) is None
    long = {"instruction": "q" * 50, "output": "a" * 50}
    assert record_to_conversation(long, max_chars=60) is None
    assert record_to_conversation(long, max_chars=0) is not None


def test_convert_records_dedupes_and_counts_skips():
    records = [_alpaca(), _alpaca(), {"instruction": "q", "output": "a"}, None]
    conversations, skipped = convert_records(records)
    assert len(conversations) == 2
    assert skipped == 2


def test_load_records_supports_json_array_and_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        array_path = os.path.join(tmp, "data.json")
        with open(array_path, "w", encoding="utf-8") as f:
            f.write('[{"instruction": "q", "output": "a"}]')
        assert load_records(array_path)[0]["instruction"] == "q"

        jsonl_path = os.path.join(tmp, "data.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write('{"instruction": "q1", "output": "a1"}\n\n')
            f.write('{"instruction": "q2", "response": "a2"}\n')
        assert len(load_records(jsonl_path)) == 2

        bad_path = os.path.join(tmp, "bad.jsonl")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("{not json}\n")
        try:
            load_records(bad_path)
        except ValueError as e:
            assert "invalid JSON" in str(e)
        else:
            raise AssertionError("expected ValueError for invalid JSON")


def test_write_jsonl_roundtrips_through_sft_loader():
    conversations, _ = convert_records([_alpaca(), _dolly()])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "chat.jsonl")
        write_chat_jsonl(conversations, path)
        loaded = load_conversations(path)
        assert len(loaded) == 2
        assert loaded[0]["messages"][0]["role"] == "user"


def test_write_text_has_no_chat_special_tokens():
    conversations, _ = convert_records([_alpaca()])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "text.txt")
        write_text(conversations, path)
        text = open(path, encoding="utf-8").read()
        assert "Cats sleep often." in text
        assert "<|" not in text
        assert conversation_to_text(conversations[0]) in text


def test_prepare_local_input_with_limit_and_shuffle():
    records = [
        {"instruction": f"q{i}", "output": f"a{i}"} for i in range(6)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "input.json")
        with open(path, "w", encoding="utf-8") as f:
            import json

            json.dump(records, f)

        first = prepare("input", input_path=path, limit=3, shuffle=True, seed=1)
        again = prepare("input", input_path=path, limit=3, shuffle=True, seed=1)
        assert len(first) == 3
        assert first == again

        try:
            prepare("input")
        except ValueError as e:
            assert "--input" in str(e)
        else:
            raise AssertionError("expected ValueError without --input")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
