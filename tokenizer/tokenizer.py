class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))                         # every unique character
        self.stoi = {c: i for i, c in enumerate(self.chars)}   # char -> id
        self.itos = {i: c for c, i in self.stoi.items()}       # id -> char
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
