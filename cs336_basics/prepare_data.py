import argparse
import pickle
from itertools import islice
from pathlib import Path

import numpy as np

from .bpe_training import iter_documents, training_bpe
from .Tokenizer import Tokenizer


END_OF_TEXT = "<|endoftext|>"


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer or encode a corpus.")
    parser.add_argument("mode", choices=("train-tokenizer", "encode"))
    parser.add_argument("input", type=Path, help="UTF-8 text file, optionally gzip-compressed")
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--output", type=Path, help="Raw little-endian uint16 token IDs")
    args = parser.parse_args()

    if args.mode == "train-tokenizer":
        if not 257 <= args.vocab_size <= 65536:
            parser.error("--vocab-size must be between 257 and 65536")
        vocab, merges = training_bpe(args.input, args.vocab_size, [END_OF_TEXT])
        for path, value in ((args.vocab, vocab), (args.merges, merges)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as file:
                pickle.dump(value, file)
        print(f"Saved {len(vocab)} tokens and {len(merges)} merges")
        return

    if args.output is None:
        parser.error("encode requires --output")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [END_OF_TEXT])
    if min(tokenizer.vocab) < 0 or max(tokenizer.vocab) > 65535:
        parser.error("vocabulary IDs must fit uint16")
    ids = tokenizer.encode_iterable(iter_documents(args.input, END_OF_TEXT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("wb") as file:
        while chunk := list(islice(ids, 65536)):
            np.asarray(chunk, dtype="<u2").tofile(file)
            count += len(chunk)
    print(f"Wrote {count} tokens to {args.output}")


if __name__ == "__main__":
    main()
