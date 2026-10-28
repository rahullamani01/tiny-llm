"""A small byte-pair-encoding tokenizer trained on a text corpus.

The vocabulary starts as the set of characters in the corpus. Each merge joins
the most frequent adjacent pair into a new token, so common words and word
fragments become single tokens.
"""
import heapq
from collections import Counter


class BPETokenizer:
    def __init__(self, chars, merges):
        self.chars = list(chars)
        self.merges = [tuple(pair) for pair in merges]

        # Expand every token id back into the text it represents.
        self.token_str = list(self.chars)
        for rank, (a, b) in enumerate(self.merges):
            if a >= len(self.token_str) or b >= len(self.token_str):
                raise ValueError(f"merge {rank} references an unknown token")
            self.token_str.append(self.token_str[a] + self.token_str[b])

        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: s for i, s in enumerate(self.token_str)}
        self.vocab_size = len(self.token_str)
        self.merge_rank = {
            pair: (rank, len(self.chars) + rank)
            for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def train(cls, text, vocab_size, max_chars=100_000, min_frequency=2):
        """Learn merges from a prefix of `text` up to `vocab_size` tokens."""
        if not text:
            raise ValueError("text must not be empty")
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")

        # Include every character in the corpus, even if the merge-training
        # prefix is shorter, so encoding the full corpus never hits an unknown id.
        chars = sorted(set(text))
        if vocab_size <= len(chars):
            return cls(chars, [])

        stoi = {c: i for i, c in enumerate(chars)}
        ids = [stoi[c] for c in text[:max_chars]]
        merges = []

        for new_id in range(len(chars), vocab_size):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            # Highest frequency wins; ties go to the lexicographically smallest pair.
            pair, frequency = max(
                counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1])
            )
            if frequency < min_frequency:
                break
            merges.append(pair)
            ids = cls._merge_all(ids, pair, new_id)

        return cls(chars, merges)

    @staticmethod
    def _merge_all(ids, pair, new_id):
        a, b = pair
        out = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def encode(self, s):
        """Apply learned merges with a linked list and a rank-ordered heap."""
        if not s:
            return []
        ids = [self.stoi[c] for c in s]
        if not self.merges:
            return ids

        n = len(ids)
        token = ids[:]
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        alive = [True] * n

        def rank_of(i, j):
            rule = self.merge_rank.get((token[i], token[j]))
            return rule[0] if rule else None

        heap = []
        for i in range(n - 1):
            rank = rank_of(i, i + 1)
            if rank is not None:
                heapq.heappush(heap, (rank, i, i + 1))

        while heap:
            rank, i, j = heapq.heappop(heap)
            if not (alive[i] and alive[j] and nxt[i] == j):
                continue
            rule = self.merge_rank.get((token[i], token[j]))
            if rule is None or rule[0] != rank:
                continue

            token[i] = rule[1]
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[i] < n:
                prev[nxt[i]] = i

            if prev[i] >= 0:
                left = prev[i]
                new_rank = rank_of(left, i)
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, left, i))
            if nxt[i] < n:
                new_rank = rank_of(i, nxt[i])
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, i, nxt[i]))

        out = []
        i = 0
        while i < n:
            out.append(token[i])
            i = nxt[i]
        return out

    def _encode_naive(self, s):
        """Reference implementation used by tests; slower but easy to verify."""
        ids = [self.stoi[c] for c in s]
        for new_id, pair in enumerate(self.merges, start=len(self.chars)):
            ids = self._merge_all(ids, pair, new_id)
        return ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def missing_chars(self, s):
        return sorted(set(s) - set(self.stoi))

    def to_dict(self):
        return {
            "type": "bpe",
            "chars": list(self.chars),
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, state):
        return cls(state["chars"], state["merges"])
