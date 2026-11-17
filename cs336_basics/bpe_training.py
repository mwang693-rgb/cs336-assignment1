import gzip
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

from .Tokenizer import PAT


def iter_documents(input_path, delimiter="<|endoftext|>"):
    """Yield chunks ending at document boundaries, retaining the delimiter."""
    open_text = gzip.open if Path(input_path).suffix == ".gz" else open
    pending = ""
    with open_text(input_path, "rt", encoding="utf-8") as file:
        while chunk := file.read(1 << 20):
            parts = (pending + chunk).split(delimiter)
            for document in parts[:-1]:
                yield document + delimiter
            pending = parts[-1]
    if pending:
        yield pending


def training_bpe(input_path, vocab_size, special_tokens) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.

    Args:
        input_path: Path to the input text file.
        vocab_size: Desired size of the vocabulary (including special tokens).
        special_tokens: List of special tokens to include in the vocabulary.

    Returns:
        A tuple containing:
            - vocab: A dictionary mapping token indices to byte strings.
            - merges: A list of tuples representing the merge operations.
    """

    special_tokens = list(dict.fromkeys(special_tokens))
    special_pattern = None
    if special_tokens:
        specials = sorted(special_tokens, key=len, reverse=True)
        special_pattern = re.compile("|".join(map(re.escape, specials)))

    # A single document delimiter can also bound the read size.
    if len(special_tokens) == 1:
        documents = iter_documents(input_path, special_tokens[0])
    else:
        open_text = gzip.open if Path(input_path).suffix == ".gz" else open
        with open_text(input_path, "rt", encoding="utf-8") as file:
            documents = (file.read(),)

    counts = Counter()
    for document in documents:
        parts = special_pattern.split(document) if special_pattern else (document,)
        for part in parts:
            counts.update(PAT.findall(part))
    originalset = Counter({
        tuple(bytes([byte]) for byte in word.encode("utf-8")): freq
        for word, freq in counts.items()
    })
    del counts

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    if vocab_size < len(vocab):
        raise ValueError("vocab_size must fit the byte vocabulary and special tokens")

    merges = []
    pair_freq = defaultdict(int)
    pair_originalset = defaultdict(set)
    for word, freq in originalset.items():
        for j in range(len(word)-1):
            pair = (word[j], word[j+1])
            pair_originalset[pair].add(word)
            if pair in pair_freq:
                pair_freq[pair] += freq
            else:
                pair_freq[pair] = freq
    while len(vocab) < vocab_size and pair_freq:
        # Find the most frequent pair, consider same frequency pairs by their dictionary order
        most_freq_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))
        if most_freq_pair[1] <= 0:
            break
        merges.append(most_freq_pair[0])
        new_token = most_freq_pair[0][0] + most_freq_pair[0][1]
        vocab[len(vocab)] = new_token
        # Replace the most frequent pair in the tokens list with the new token
        for word in list(pair_originalset[most_freq_pair[0]]):
            freq = originalset[word]
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == most_freq_pair[0]:
                                # --- 1. 捕捉【前受害者】: word[i-1] 和 word[i] 组成的 pair ---
                    # 执行合并
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            originalset[tuple(new_word)] += freq
            del originalset[word]
            for j in range(len(new_word)-1):
                pair = (new_word[j], new_word[j+1])
                pair_originalset[pair].add(tuple(new_word))
                if pair in pair_freq:
                    pair_freq[pair] += freq
                else:
                    pair_freq[pair] = freq
            for j in range(len(word)-1):
                pair = (word[j], word[j+1])
                pair_originalset[pair].discard(word)
                pair_freq[pair] -= freq
            
            # pair_originalset[new_next_pair].add(tuple(new_word) + (word[i] + word[i+1],) + word[i+2:])
            # pair_originalset[next_pair].discard(word)
        del pair_originalset[most_freq_pair[0]]
        del pair_freq[most_freq_pair[0]]
            
        # slow method
        # for word in list(originalset.key()):
        #     if most_freq_pair[0][0] in word and most_freq_pair[0][1] in word:
        #             new_word = []
        #             i = 0
        #             while i < len(word):
        #                 if i < len(word) - 1 and (word[i], word[i+1]) == most_freq_pair[0]:
        #                     new_word.append(new_token)
        #                     i += 2
        #                 else:
        #                     new_word.append(word[i])
        #                     i += 1
        #             new_originalset[tuple(new_word)] += freq
        #             del originalset[word]
        #     else:
        #     new_word = word.replace(most_freq_pair[0][0] + most_freq_pair[0][1], new_token)
        #     new_originalset[new_word] += freq
        # originalset = new_originalset
        # replace is not working correctly, we only replace the pair self we need to replace the near pairs, we can use regex to replace the pair
    return vocab, merges







            