"""Run every test module without needing pytest: python -m tests.run_all"""
import importlib

MODULES = (
    "tests.test_model",
    "tests.test_generation",
    "tests.test_training",
    "tests.test_bpe",
    "tests.test_cache",
    "tests.test_rope",
    "tests.test_chat",
    "tests.test_server",
    "tests.test_prepare_chat",
    "tests.test_tokenizer_baseline",
    "tests.test_chat_experiment",
    "tests.test_byte",
    "tests.test_evaluate_chat",
)


def main():
    failures = 0
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception as e:  # keep going so one failure shows them all
                failures += 1
                print(f"FAIL: {module_name}.{name}: {e}")
            else:
                print(f"ok: {module_name}.{name}")

    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("all tests passed")


if __name__ == "__main__":
    main()
