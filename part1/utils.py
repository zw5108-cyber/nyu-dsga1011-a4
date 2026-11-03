"""Deterministic, label-independent spelling transformation for Q2/Q3."""
import hashlib
import random
import re


def example_transform(example):
    return {**example, "text": example["text"].lower()}


def custom_transform(example):
    """Lowercase; transpose one unequal internal adjacent pair in 40% of long words.

    Alphabetic words of length >=5 are eligible. First and last letters stay
    unchanged; short words (including no/not) and punctuation are preserved.
    A SHA-256 hash of the original text seeds a private RNG, so repeated calls
    produce exactly the same text regardless of order or train/eval process.
    """
    original = example["text"]
    rng = random.Random(int.from_bytes(hashlib.sha256(original.encode("utf-8")).digest()[:8], "big"))

    def transpose(match):
        word = match.group(0)
        if len(word) < 5 or rng.random() >= 0.4:
            return word
        positions = [i for i in range(1, len(word) - 2) if word[i] != word[i + 1]]
        if not positions:
            return word
        i = rng.choice(positions)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]

    return {**example, "text": re.sub(r"[a-z]+", transpose, original.lower())}
